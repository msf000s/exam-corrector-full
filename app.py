from flask import Flask, request, jsonify, send_from_directory
import os
import json
import base64
import requests
import google.generativeai as genai
from datetime import datetime

app = Flask(__name__, static_folder='static')
os.makedirs('static', exist_ok=True)

# ⚙️ تحميل الإجابات الصحيحة
def load_answer_key():
    try:
        if os.path.exists('answer_key.json'):
            with open('answer_key.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('answers', [])
    except:
        pass
    return ['B', 'C', 'A', 'D', 'B', 'A', 'C', 'D', 'A', 'B']

def save_answer_key(answers):
    try:
        with open('answer_key.json', 'w', encoding='utf-8') as f:
            json.dump({'answers': answers, 'updated': datetime.now().isoformat()}, f, ensure_ascii=False)
    except:
        pass

ANSWER_KEY = load_answer_key()

# 🔑 ضع مفتاح API الخاص بك هنا
genai.configure(api_key='YOUR_API_KEY_HERE')
model = genai.GenerativeModel('gemini-1.5-flash')  # أو gemini-pro-vision

def extract_answers_from_image(image_bytes, num_questions=10, options_per_q=4):
    """
    استخدام Gemini لقراءة الصورة واستخراج الإجابات
    """
    try:
        # تحويل الصورة إلى تنسيق يفهمه Gemini
        image = genai.upload_file(image_bytes)
        
        # مطالبة Gemini بقراءة الإجابات
        prompt = f"""
        أنت مساعد ذكي مسؤول عن قراءة نموذج إجابة طالب.
        الصورة المرفقة تحتوي على {num_questions} سؤال، وكل سؤال يحتوي على {options_per_q} خيارات (A، B، C، D).
        قم بتحليل الصورة وتحديد الخيار المظلل أو المحدد في كل سؤال.
        يجب أن تُرجع النتيجة كمصفوفة من 10 عناصر، مثل:
        ["A", "B", "فراغ", "D", "C", "A", "فراغ", "B", "C", "A"]
        استخدم "فراغ" إذا لم يتم تحديد أي خيار.
        لا تكتب أي شيء آخر، فقط المصفوفة.
        """
        
        response = model.generate_content([prompt, image])
        result = response.text.strip()
        
        # تحويل النص إلى مصفوفة
        import ast
        try:
            answers = ast.literal_eval(result)
            if isinstance(answers, list) and len(answers) == num_questions:
                return answers
            else:
                raise ValueError("النتيجة ليست مصفوفة صحيحة")
        except:
            # fallback: إذا فشل التحليل، نُرجع فراغات
            return ["فراغ"] * num_questions

    except Exception as e:
        print(f"خطأ في Gemini API: {str(e)}")
        return ["فراغ"] * num_questions

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/answer_key', methods=['GET', 'POST'])
def handle_answer_key():
    if request.method == 'GET':
        return jsonify({
            "answers": ANSWER_KEY,
            "count": len(ANSWER_KEY)
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            new_answers = data.get('answers', [])
            
            if new_answers:
                global ANSWER_KEY
                ANSWER_KEY = new_answers
                save_answer_key(new_answers)
                
                return jsonify({
                    "success": True,
                    "message": f"تم تحديث {len(ANSWER_KEY)} إجابة صحيحة",
                    "answers": ANSWER_KEY
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "لم يتم تقديم إجابات صحيحة"
                }), 400
                
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"خطأ في تحديث الإجابات: {str(e)}"
            }), 500

@app.route('/api/correct', methods=['POST'])
def correct():
    """تصحيح ورقة الإجابة باستخدام Gemini API"""
    try:
        # الحصول على المعلمات
        num_questions = request.form.get('num_questions', default=len(ANSWER_KEY), type=int)
        options_per_q = request.form.get('options_per_q', default=4, type=int)
        
        # معالجة ملف الصورة
        if 'image' not in request.files:
            return jsonify({"error": "لم يتم تقديم صورة"}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "لم يتم اختيار ملف"}), 400

        # قراءة الصورة كـ bytes
        image_bytes = file.read()

        # استخراج الإجابات باستخدام Gemini
        student_answers = extract_answers_from_image(image_bytes, num_questions, options_per_q)
        
        # حساب النتائج
        correct_count = sum(1 for s, c in zip(student_answers, ANSWER_KEY[:num_questions]) if s == c)
        wrong_count = num_questions - correct_count
        percentage = round((correct_count / num_questions) * 100) if num_questions > 0 else 0
        
        # تحضير الاستجابة
        response = {
            "success": True,
            "answers": student_answers,
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "percentage": percentage,
            "total_questions": num_questions,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print("❌ خطأ في /api/correct:", str(e))
        return jsonify({
            "success": False,
            "error": "خطأ داخلي في معالجة الصورة"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "answer_key_count": len(ANSWER_KEY)
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🚀 بدء تشغيل تطبيق تصحيح الاختبارات...")
    print(f"📊 عدد الإجابات الصحيحة: {len(ANSWER_KEY)}")
    print(f"🌐 الخادم يعمل على: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
