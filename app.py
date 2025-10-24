from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
from PIL import Image
import json
import logging

# --- الإعدادات الأساسية ---
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
# نخبر فلاسك أن الملفات الثابتة (مثل index.html) موجودة في مجلد "static"
app = Flask(__name__, static_folder='static')

# 3. المسارات (Routes)

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    logging.info("يتم طلب الواجهة الأمامية (index.html)...")
    return send_from_directory(app.static_folder, 'index.html')

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
        num_questions = int(request.form.get('num_questions', 10))
        options_per_q = int(request.form.get('options_per_q', 4))

        logging.info(f"يتم معالجة صورة لـ {num_questions} سؤال (أسئلة) مع {options_per_q} خيار (خيارات).")
        img = Image.open(image_file.stream)

        options_str = "A, B, C, D" if options_per_q == 4 else "A, B, C, D, E"
        
        # --- الـ Prompt (الأمر) ---
        prompt = f"""
أنت مساعد ذكاء اصطناعي متخصص في قراءة نماذج الإجابات.
هذه صورة لنموذج إجابة اختبار من متعدد.
عدد الأسئلة: {num_questions}
عدد الخيارات لكل سؤال: {options_per_q} ({options_str})

مهم جداً: قم بالرد فقط باستخدام قائمة JSON تحتوي على الإجابات.
مثال: ["A", "B", "Blank", "C"]
استخدم القيمة "Blank" (بالضبط بهذه الحروف) إذا لم يتم تظليل أي خيار للسؤال.
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
            logging.info(f"الاستجابة الأولية من Gemini: >>>{raw_text}<<<")
        except Exception as text_err:
            logging.error(f"خطأ أثناء محاولة قراءة response.text: {text_err}")
            logging.warning(f"ربما تم حظر الاستجابة؟ Response object: {response}")
            return jsonify({"success": False, "error": f"Error accessing response text from Gemini. Check safety settings. Details: {response.prompt_feedback}"}), 500

        if not raw_text:
            logging.error("Gemini لم يرجع أي نص (raw_text is empty).")
            return jsonify({"success": False, "error": "Did not receive text content from Gemini"}), 500
        
        answers = None
        cleaned_text = raw_text
        try:
            # محاولة إزالة علامات Markdown إذا وجدت
            if raw_text.startswith("```json"):
                cleaned_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                cleaned_text = raw_text[3:-3].strip()
            
            answers = json.loads(cleaned_text)
            logging.info(f"تم تحليل الإجابات بنجاح: {answers}")
        
        except json.JSONDecodeError as json_err:
            logging.error(f"فشل تحليل JSON: {cleaned_text}")
            logging.error(f"JSONDecodeError: {json_err}")
            logging.error(f"(الاستجابة الأولية كانت: {raw_text})")
            return jsonify({
                "success": False, 
                "error": "Unexpected format from Gemini (JSON decode error)",
                "details": cleaned_text
            }), 500
        
        if not isinstance(answers, list):
            logging.error(f"البيانات المستلمة ليست قائمة (list), بل هي: {type(answers)}")
            return jsonify({"success": False, "error": "The response was not a list."}), 500

        if len(answers) != num_questions:
            logging.warning(f"عدد الإجابات ({len(answers)}) لا يطابق عدد الأسئلة ({num_questions}).")
            return jsonify({
                "success": False,
                "error": f"Number of answers ({len(answers)}) does not match number of questions ({num_questions})",
                "details": answers
            }), 500

        return jsonify({"success": True, "answers": answers})

    except Exception as e:
        logging.error(f"حدث خطأ فادح في /api/correct: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

# 4. نقطة بدء تشغيل الخادم
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"بدء تشغيل الخادم على http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
