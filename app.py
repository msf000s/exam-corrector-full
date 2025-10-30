from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
from PIL import Image
import json
import logging
from io import BytesIO

# --- الإعدادات الأساسية (بدون تغيير) ---
logging.basicConfig(level=logging.INFO)

# 1. إعدادات نموذج Gemini
try:
    # --- تأكد من ضبط هذا المتغير في بيئة التشغيل لديك ---
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("لم يتم العثور على مفتاح GEMINI_API_KEY في متغيرات البيئة")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # --- (التعديل) استخدام اسم الموديل الصحيح الذي ذكرته ---
    model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("تم تهيئة Gemini بنجاح باستخدام 'gemini-2.5-flash'.")

    # --- إعدادات الإنشاء (للحصول على JSON ثابت) ---
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json", # طلب JSON مباشرة
    }

    # --- إعدادات السلامة (لتجنب حظر صور الأوراق) ---
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

except Exception as e:
    logging.error(f"خطأ أثناء تهيئة Gemini: {e}")
    logging.error("تأكد من صحة المفتاح 'GEMINI_API_KEY' وأن الموديل 'gemini-2.5-flash' متاح.")
    model = None
    generation_config = {}
    safety_settings = []

# 2. إعداد خادم Flask
# نستخدم '..' لتحديد أن المجلد الثابت هو المجلد الحالي
app = Flask(__name__, static_folder='.', static_url_path='')

# 3. المسارات (Routes)

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    logging.info("يتم طلب الواجهة الأمامية (index.html)...")
    # استخدام send_file إذا كان الملف في المجلد الحالي
    return send_from_directory('.', 'index.html')

# API endpoint for processing the answer sheet
@app.route('/api/correct', methods=['POST'])
def correct_answers():
    logging.info("تم استلام طلب تصحيح جديد... /api/correct")
    if model is None:
        logging.error("الموديل غير مهيأ. لا يمكن المتابعة.")
        return jsonify({"success": False, "error": "Gemini model is not initialized. Check API key."}), 500
        
    try:
        if 'image' not in request.files:
            logging.warning("الطلب لا يحتوي على ملف 'image'.")
            return jsonify({"success": False, "error": "No image was uploaded"}), 400

        image_file = request.files['image']
        
        # (تعديل 1) معالجة آمنة للمتغيرات الرقمية
        try:
            num_questions = int(request.form.get('num_questions', 10))
            options_per_q = int(request.form.get('options_per_q', 4))
        except ValueError:
            logging.warning("قيمة غير صالحة لعدد الأسئلة أو الخيارات.")
            return jsonify({"success": False, "error": "Invalid question or options count."}), 400

        logging.info(f"يتم معالجة صورة لـ {num_questions} سؤال (أسئلة) مع {options_per_q} خيار (خيارات).")
        img = Image.open(image_file.stream)

        options_letters = [chr(65 + i) for i in range(options_per_q)]
        options_str = ", ".join(options_letters)
        
        # --- الـ Prompt (الأمر) ---
        prompt = f"""
أنت مساعد ذكاء اصطناعي متخصص في قراءة نماذج الإجابات.
هذه صورة لنموذج إجابة اختبار من متعدد.
عدد الأسئلة: {num_questions}
الخيارات الممكنة هي: {options_str}

مهم جداً: قم بالرد فقط باستخدام قائمة JSON تحتوي على الإجابات.
مثال: ["A", "B", "Blank", "C"]
استخدم القيمة "Blank" (بالضبط بهذه الحروف B-l-a-n-k) إذا لم يتم تظليل أي خيار للسؤال.
يجب أن تحتوي القائمة على {num_questions} عنصر (إجابة) بالضبط.
لا تقم بكتابة أي شروحات أو نصوص إضافية. الرد يجب أن يكون قائمة JSON فقط.
""".strip()

        # إرسال الطلب إلى Gemini مع الإعدادات الجديدة
        response = model.generate_content(
            [prompt, img],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        raw_text = ""
        try:
            raw_text = response.text.strip()
        except Exception as text_err:
            logging.error(f"خطأ أثناء محاولة قراءة response.text: {text_err}")
            return jsonify({"success": False, "error": f"Error accessing response text from Gemini. Check safety settings. Details: {response.prompt_feedback}"}), 500

        if not raw_text:
            logging.error("Gemini لم يرجع أي نص (raw_text is empty).")
            return jsonify({"success": False, "error": "Did not receive text content from Gemini"}), 500
        
        answers = None
        cleaned_text = raw_text
        try:
            # محاولة إزالة علامات Markdown إذا وجدت
            if raw_text.startswith("```json"):
                cleaned_text = raw_text[7:].rstrip("`").strip()
            elif raw_text.startswith("```"):
                cleaned_text = raw_text[3:].rstrip("`").strip()
            
            answers = json.loads(cleaned_text)
            logging.info(f"تم تحليل الإجابات بنجاح: {answers}")
        
        except json.JSONDecodeError as json_err:
            logging.error(f"فشل تحليل JSON: {cleaned_text}")
            logging.error(f"JSONDecodeError: {json_err}")
            logging.error(f"(الاستجابة الأولية كانت: {raw_text})")
            return jsonify({
                "success": False, 
                "error": "Unexpected JSON format from Gemini",
                "details": cleaned_text
            }), 500
        
        if not isinstance(answers, list):
            logging.error(f"البيانات المستلمة ليست قائمة (list), بل هي: {type(answers)}")
            return jsonify({"success": False, "error": "The response was not a list."}), 500

        if len(answers) != num_questions:
            logging.warning(f"(تعديل 2) عدد الإجابات ({len(answers)}) لا يطابق عدد الأسئلة ({num_questions}).")
            # (تعديل 2) رسالة خطأ أوضح للمستخدم
            return jsonify({
                "success": False,
                "error": f"Number of answers ({len(answers)}) does not match expected questions ({num_questions}). Check model configuration or image clarity.",
                "details": answers
            }), 500

        # (تعديل 3) تحويل الإجابات للتأكد من أنها تتطابق مع التنسيق المطلوب في JS
        # يجب أن يكون Gemini قادراً على قراءة 'Blank' ولكن هذا التحويل البسيط يضمن النظافة
        final_answers = [
            ans.upper().strip() if isinstance(ans, str) else 'Blank'
            for ans in answers
        ]

        return jsonify({"success": True, "answers": final_answers})

    except Exception as e:
        logging.error(f"حدث خطأ فادح في /api/correct: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

# 4. نقطة بدء تشغيل الخادم
if __name__ == '__main__':
    # (تعديل 4) إضافة مسار 'static' للتشغيل المحلي المباشر
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app.static_folder = current_dir
    app.static_url_path = '/'
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"بدء تشغيل الخادم على http://127.0.0.1:{port}")
    # تعطيل debug للتوافق مع gunicorn على render
    app.run(host='0.0.0.0', port=port, debug=False)
