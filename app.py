from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
from PIL import Image
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
# نخبر فلاسك أن الملفات الثابتة (مثل index.html) موجودة في مجلد "static"
app = Flask(__name__, static_folder='static')

# Configure Gemini API key from environment variable
try:
    # --- تأكد من ضبط هذا المتغير في بيئة التشغيل لديك ---
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    if not GEMINI_API_KEY:
        raise ValueError("لم يتم العثور على مفتاح GEMINI_API_KEY في متغيرات البيئة")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # تحديد الموديل
    model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info("تم تهيئة Gemini بنجاح باستخدام 'gemini-1.5-flash'.")

except Exception as e:
    logging.error(f"خطأ أثناء تهيئة Gemini: {e}")
    logging.error("تأكد من صحة المفتاح 'GEMINI_API_KEY' وأن الموديل 'gemini-1.5-flash' متاح.")
    model = None

# Route to serve the frontend (index.html)
# هذا هو الرابط الرئيسي الذي يفتح التطبيق في المتصفح
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
        
        # --- تحسين الـ Prompt ---
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

        # إرسال الطلب إلى Gemini
        response = model.generate_content([prompt, img])

        # --- Enhanced Logging ---
        if not response:
             logging.error("Received NO response object from Gemini.")
             return jsonify({"success": False, "error": "Did not receive response object from Gemini"}), 500
        
        raw_text = "" # Initialize raw_text
        try:
             # Log the raw response text if available
             raw_text = response.text.strip()
             logging.info(f"الاستجابة الأولية من Gemini: >>>{raw_text}<<<") # Added delimiters
        except Exception as text_err:
             # Log if accessing response.text itself causes an error
             logging.error(f"خطأ أثناء محاولة قراءة response.text: {text_err}")
             logging.error(f"Response object received: {response}") # Log the whole response object
             return jsonify({"success": False, "error": "Error accessing response text from Gemini"}), 500
        # --- End Enhanced Logging ---

        # Check if raw_text is empty *after* logging and attempting to read response.text
        if not raw_text: 
            logging.error("Gemini لم يرجع أي نص (raw_text is empty).")
            # Consider if parts might contain the data even if text is empty
            if response.parts:
                 logging.warning(f"Response.text was empty, but response.parts exist: {response.parts}")
                 # Attempt to extract from parts if necessary (more complex)
            return jsonify({"success": False, "error": "Did not receive text content from Gemini"}), 500
        
        # تنظيف الاستجابة من أي علامات Markdown
        cleaned_text = raw_text # Start with the raw text
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:-3].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:-3].strip()

        # التأكد أن الاستجابة هي قائمة JSON
        if not cleaned_text.startswith("[") or not cleaned_text.endswith("]"):
            logging.error(f"الاستجابة بعد التنظيف ليست بتنسيق JSON list: {cleaned_text}")
            logging.error(f"(الاستجابة الأولية كانت: {raw_text})") # Log original for comparison
            return jsonify({
                "success": False, 
                "error": "Unexpected format from Gemini (not a list after cleaning)",
                "details": cleaned_text # Return cleaned text for debugging
            }), 500

        answers = None
        try:
            answers = json.loads(cleaned_text) # Use cleaned_text for parsing
            logging.info(f"تم تحليل الإجابات بنجاح: {answers}")
        except json.JSONDecodeError as json_err:
            logging.error(f"فشل تحليل JSON: {cleaned_text}")
            logging.error(f"JSONDecodeError: {json_err}") # Log the specific JSON error
            logging.error(f"(الاستجابة الأولية كانت: {raw_text})")
            return jsonify({
                "success": False, 
                "error": "Unexpected format from Gemini (JSON decode error)",
                "details": cleaned_text
            }), 500
        
        if not isinstance(answers, list):
            logging.error(f"البيانات المستلمة ليست قائمة (list), بل هي: {type(answers)}")
            return jsonify({"success": False, "error": "The response was not a list."}), 500

        # التحقق من تطابق عدد الأسئلة
        if len(answers) != num_questions:
            logging.warning(f"عدد الإجابات ({len(answers)}) لا يطابق عدد الأسئلة ({num_questions}).")
            return jsonify({
                "success": False,
                "error": f"Number of answers ({len(answers)}) does not match number of questions ({num_questions})",
                "details": answers
            }), 500

        # إرجاع الاستجابة الناجحة
        return jsonify({"success": True, "answers": answers})

    except Exception as e:
        logging.error(f"حدث خطأ فادح في /api/correct: {e}", exc_info=True) # Log full traceback
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

# Main entry point to run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"بدء تشغيل الخادم على http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

