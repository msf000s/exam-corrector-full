from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import imutils
import traceback
import json
from datetime import datetime
import base64

app = Flask(__name__, static_folder='static')

# إنشاء المجلدات إذا لم تكن موجودة
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

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

# 🔧 دالة ترتيب النقاط لتحويل المنظور
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_bubbles(warped_gray):
    """كشف الفقاعات في الصورة"""
    try:
        # تحسين الصورة
        blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # تنظيف الصورة
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # العثور على الكنتورات
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        bubbles = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100 or area > 5000:
                continue
                
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:
                continue
                
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            
            if 0.7 <= aspect_ratio <= 1.3:
                bubbles.append({
                    'contour': c,
                    'center': (x + w//2, y + h//2),
                    'radius': min(w, h) // 2,
                    'bbox': (x, y, w, h)
                })
        
        return bubbles, None
        
    except Exception as e:
        print(f"خطأ في detect_bubbles: {str(e)}")
        return [], None

def extract_answers(warped_gray, num_questions=10, options_per_q=4):
    """استخراج الإجابات من الصورة"""
    try:
        bubbles, _ = detect_bubbles(warped_gray)
        
        if not bubbles:
            return ["فراغ"] * num_questions
        
        # ترتيب الفقاعات
        bubbles_sorted = sorted(bubbles, key=lambda b: b['center'][1])
        
        # تجميع في صفوف
        rows = []
        current_row = [bubbles_sorted[0]]
        row_threshold = bubbles_sorted[0]['radius'] * 2
        
        for bubble in bubbles_sorted[1:]:
            if abs(bubble['center'][1] - current_row[0]['center'][1]) <= row_threshold:
                current_row.append(bubble)
            else:
                current_row.sort(key=lambda b: b['center'][0])
                rows.append(current_row)
                current_row = [bubble]
        
        if current_row:
            current_row.sort(key=lambda b: b['center'][0])
            rows.append(current_row)
        
        # استخراج الإجابات
        answers = []
        for i, row in enumerate(rows):
            if i >= num_questions:
                break
                
            if len(row) < options_per_q:
                answers.append("فراغ")
                continue
            
            filled_bubbles = []
            for j, bubble in enumerate(row[:options_per_q]):
                x, y, w, h = bubble['bbox']
                roi = warped_gray[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                filled_ratio = cv2.countNonZero(binary_roi) / (w * h)
                filled_bubbles.append((j, filled_ratio))
            
            if not filled_bubbles:
                answers.append("فراغ")
                continue
            
            filled_bubbles.sort(key=lambda x: x[1], reverse=True)
            best_bubble_idx, best_ratio = filled_bubbles[0]
            
            if best_ratio > 0.3:
                options = ['A', 'B', 'C', 'D', 'E'][:options_per_q]
                answers.append(options[best_bubble_idx])
            else:
                answers.append("فراغ")
        
        while len(answers) < num_questions:
            answers.append("فراغ")
            
        return answers[:num_questions]
        
    except Exception as e:
        print(f"خطأ في extract_answers: {str(e)}")
        return ["فراغ"] * num_questions

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/answer_key', methods=['GET', 'POST'])
def handle_answer_key():
    """إدارة الإجابات الصحيحة"""
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
    """تصحيح ورقة الإجابة"""
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
        
        # قراءة الصورة مباشرة من الذاكرة
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "فشل في قراءة الصورة"}), 400
        
        # المعالجة
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # العثور على الكنتورات
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        docCnt = None
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    docCnt = approx
                    break
        
        # تحويل منظور الصورة
        if docCnt is not None:
            warped = four_point_transform(gray, docCnt.reshape(4, 2))
        else:
            warped = gray
        
        # استخراج الإجابات
        student_answers = extract_answers(warped, num_questions, options_per_q)
        
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
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "خطأ داخلي في معالجة الصورة"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """فحص حالة الخادم"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "answer_key_count": len(ANSWER_KEY)
    })

# 🔧 إصلاح مشكلة CORS إذا لزم الأمر
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
