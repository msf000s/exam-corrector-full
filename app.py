from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import os
from PIL import Image
import json

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Configure Gemini API key from environment variable
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_AI_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_AI_KEY is not set in the environment variables")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # --- التغيير النهائي هنا ---
    # استخدام اسم نموذج صحيح ومتاح من القائمة التي تم استخراجها
    model = genai.GenerativeModel('gemini-2.5-flash')

except Exception as e:
    print(f"Error during Gemini initialization: {e}")
    model = None

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint for processing the answer sheet
@app.route('/api/correct', methods=['POST'])
def correct_answers():
    if model is None:
        return jsonify({"success": False, "error": "Gemini model is not initialized. Check API key."}), 500
        
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image was uploaded"}), 400

        image_file = request.files['image']
        num_questions = int(request.form.get('num_questions', 10))
        options_per_q = int(request.form.get('options_per_q', 4))

        img = Image.open(image_file.stream)

        options_str = "A, B, C, D" if options_per_q == 4 else "A, B, C, D, E"
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

        response = model.generate_content([prompt, img])

        if not response.text:
            return jsonify({"success": False, "error": "Did not receive a response from Gemini"}), 500

        raw_text = response.text.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        answers = None
        try:
            cleaned_text = raw_text.replace('"Blank"', '"فراغ"')
            answers = json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(f"JSON decoding failed for: {raw_text}")
            return jsonify({
                "success": False, 
                "error": "Unexpected format from Gemini",
                "details": raw_text
            }), 500
        
        if not isinstance(answers, list):
            return jsonify({"success": False, "error": "The response was not a list."}), 500

        if len(answers) != num_questions:
            return jsonify({
                "success": False,
                "error": f"Number of answers ({len(answers)}) does not match number of questions ({num_questions})"
            }), 500

        return jsonify({"success": True, "answers": answers})

    except Exception as e:
        print(f"An error occurred in /api/correct: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Main entry point to run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))


