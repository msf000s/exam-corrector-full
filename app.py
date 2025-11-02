from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from google.generativeai import types
import os
from PIL import Image
import json
import logging
import base64
from io import BytesIO
import sqlite3
import atexit # لإغلاق الاتصال بقاعدة البيانات عند خروج التطبيق

# --- الإعدادات الأساسية (بدون تغيير) ---
logging.basicConfig(level=logging.INFO)

# 1. إعدادات نموذج Gemini
# ... (كود إعداد Gemini الحالي) ...

try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("لم يتم العثور على مفتاح GEMINI_API_KEY في متغيرات البيئة")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("تم تهيئة Gemini بنجاح.")

    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json", 
    }

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
app = Flask(__name__, static_folder='static', static_url_path='/static') 

# ----------------------------------------------------
# 💾 دوال إدارة قاعدة البيانات SQLite الجديدة 
# ----------------------------------------------------

DATABASE = 'omr_data.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """إنشاء الجداول عند بدء التشغيل."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. جدول الطلاب
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            latest_result_json TEXT 
        );
    """)
    
    # 2. جدول نماذج الإجابة
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS answer_keys (
            name TEXT PRIMARY KEY,
            subject TEXT,
            exam_name TEXT,
            exam_date TEXT,
            answers_json TEXT NOT NULL
        );
    """)
    
    # 3. جدول سجلات النتائج المفصلة (Detailed Log)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            student_name TEXT,
            student_class TEXT,
            exam_name TEXT,
            exam_date TEXT,
            score_summary TEXT,
            percentage TEXT,
            grade TEXT,
            answers_log_json TEXT NOT NULL
        );
    """)
    
    conn.commit()
    conn.close()
    logging.info("تم تهيئة قاعدة بيانات SQLite بنجاح.")

# تنفيذ تهيئة قاعدة البيانات عند بدء تشغيل التطبيق
init_db()

# ----------------------------------------------------
# 3. المسارات (Routes)
# ----------------------------------------------------

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint for processing the answer sheet
@app.route('/api/correct', methods=['POST'])
def correct_answers():
    # ... (كود التصحيح الحالي) ...
    # هنا لا يلزم أي تغيير، حيث أن هذه الدالة ترسل وتستقبل البيانات من Gemini فقط.
    # التغيير سيتم في الدوال التي تستبدل localStorage (مثل حفظ/تحميل الطلاب والنماذج)

    logging.info("تم استلام طلب تصحيح جديد... /api/correct")
    if model is None:
        return jsonify({"success": False, "error": "Gemini model is not initialized. Check API key."}), 500
        
    try:
        data = request.get_json()
        image_base64 = data.get('image_base64')
        num_questions = int(data.get('num_questions', 10))
        options_per_q = int(data.get('options_per_q', 4))
        correct_answers_list = data.get('correct_answers', []) 
        
        if not image_base64:
            return jsonify({"success": False, "error": "No image data uploaded"}), 400

        image_bytes = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_bytes))

        is_key_scan = len(correct_answers_list) == 0
        options_letters = [chr(65 + i) for i in range(options_per_q)]
        options_str = ", ".join(options_letters)
        
        # --- بناء الـ Prompt (الأمر) ---
        if is_key_scan:
            prompt_role = "أنت مساعد متخصص في قراءة نماذج الإجابات الصحيحة."
            prompt_task = f"""
                مهمتك هي قراءة الإجابات المظللة في ورقة الإجابة هذه وتحديد الإجابة الصحيحة لكل سؤال.
                التركيز يجب أن يكون على منطقة الإجابات فقط لتجنب أي تشويش.
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
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8000), debug=False)
