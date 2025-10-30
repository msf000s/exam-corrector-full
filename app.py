from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from google.generativeai import types
import os
from PIL import Image
import json
import logging
import base64
from io import BytesIO

# --- الإعدادات الأساسية ---
logging.basicConfig(level=logging.INFO)

# 1. إعدادات نموذج Gemini
try:
    # --- تأكد من ضبط هذا المتغير في بيئة التشغيل لديك ---
    # يجب ضبط GEMINI_API_KEY كمتغير بيئة على منصة Render
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        # سيتم التعامل مع هذا الخطأ أثناء التهيئة، وهو السلوك المتوقع على Render إذا لم يتم ضبط المتغير
        raise ValueError("لم يتم العثور على مفتاح GEMINI_API_KEY في متغيرات البيئة")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("تم تهيئة Gemini بنجاح.")

    # إعدادات الإنشاء لضمان الحصول على JSON
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json", 
    }

    # إعدادات السلامة (للسماح بمعالجة صور الأوراق)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

except Exception as e:
    logging.error(f"خطأ أثناء تهيئة Gemini: {e}")
    model = None
    generation_config = {}
    safety_settings = []

# 2. إعداد خادم Flask
# static_folder='.' لتمكين Render من العثور على index.html في جذر المشروع
app = Flask(__name__, static_folder='.', static_url_path='')

# 3. المسارات (Routes)

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# API endpoint for processing the answer sheet
@app.route('/api/correct', methods=['POST'])
def correct_answers():
    logging.info("تم استلام طلب تصحيح جديد... /api/correct")
    if model is None:
        return jsonify({"success": False, "error": "Gemini model is not initialized. Check API key."}), 500
        
    try:
        # استلام البيانات بتنسيق JSON من الواجهة الأمامية
        data = request.get_json()
        image_base64 = data.get('image_base64')
        num_questions = int(data.get('num_questions', 10))
        options_per_q = int(data.get('options_per_q', 4))
        correct_answers_list = data.get('correct_answers', []) 
        
        if not image_base64:
            return jsonify({"success": False, "error": "No image data uploaded"}), 400

        # فك تشفير الصورة
        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))

        is_key_scan = len(correct_answers_list) == 0
        options_letters = [chr(65 + i) for i in range(options_per_q)]
        options_str = ", ".join(options_letters)
        
        # --- بناء الـ Prompt (الأمر) المُحسَّن للتركيز ---
        if is_key_scan:
            prompt_role = "أنت مساعد متخصص في قراءة نماذج الإجابات الصحيحة."
            prompt_task = f"""
                مهمتك هي قراءة الإجابات المظللة في ورقة الإجابة هذه وتحديد الإجابة الصحيحة لكل سؤال.
                التركيز يجب أن يكون على منطقة الإجابات فقط لتجاهل أي تشويش.
                يرجى الرد بقائمة JSON تحتوي على إجابة واحدة لكل سؤال من {num_questions} سؤال.
                استخدم القيمة "Blank" إذا لم يتم تظليل أي شيء.
            """
        else:
            prompt_role = "أنت مساعد متخصص في قراءة إجابات الطلاب ومقارنتها بمفتاح الإجابة."
            prompt_task = f"""
                مهمتك هي قراءة تظليل الطالب لهذه الورقة.
                التركيز يجب أن يكون على منطقة الإجابات فقط لتجنب أي تشويش.
                الإجابات الصحيحة هي: {', '.join(correct_answers_list)}
                يرجى الرد بقائمة JSON تحتوي على إجابة الطالب لكل سؤال من {num_questions} سؤال.
                استخدم القيمة "Blank" إذا لم يتم تظليل أي شيء.
            """

        prompt = f"""
            {prompt_role}
            هذه صورة لنموذج إجابة اختبار من متعدد.
            عدد الأسئلة: {num_questions}
            الخيارات الممكنة هي: {options_str}

            {prompt_task}
            
            مثال للنمط المطلوب (يجب أن يحتوي على {num_questions} عنصر بالضبط):
            ["A", "Blank", "C", "D", ... ]
            
            لا تقم بكتابة أي شروحات أو نصوص إضافية. الرد يجب أن يكون قائمة JSON فقط.
        """.strip()

        # إرسال الطلب إلى Gemini
        response = model.generate_content(
            [prompt, img],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        raw_text = response.text.strip()
        
        # تنظيف وإعادة محاولة تحليل JSON
        cleaned_text = raw_text
        if raw_text.startswith("```json"):
            cleaned_text = raw_text[7:].rstrip("`").strip()
        elif raw_text.startswith("```"):
            cleaned_text = raw_text[3:].rstrip("`").strip()
        
        answers = json.loads(cleaned_text)

        if not isinstance(answers, list) or len(answers) != num_questions:
            return jsonify({
                "success": False,
                "error": f"النموذج أعاد {len(answers)} إجابة، لكن المتوقع {num_questions}. يرجى تحسين الإضاءة أو وضوح الصورة.",
            }), 500

        # تحويل الإجابات للتأكد من أنها تتطابق مع التنسيق المطلوب في JS (مثل Blank)
        final_answers = [str(ans).upper().strip() if isinstance(ans, str) else 'Blank' for ans in answers]

        return jsonify({"success": True, "answers": final_answers})

    except json.JSONDecodeError:
        logging.error(f"JSON Decode Error. Gemini Response: {raw_text if 'raw_text' in locals() else 'N/A'}")
        return jsonify({"success": False, "error": "فشل في قراءة استجابة النموذج (تنسيق JSON غير صحيح)."}), 500
    except ValueError as ve:
        return jsonify({"success": False, "error": f"Invalid input data: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"General Error in /api/correct: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

# 4. نقطة بدء تشغيل الخادم
if __name__ == '__main__':
    # إعدادات التشغيل المحلي
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app.static_folder = current_dir
    app.static_url_path = '/'
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
