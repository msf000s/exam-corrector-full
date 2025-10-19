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

ANSWER_KEY = load_answer_key()

def order_points(pts):
    """ترتيب النقاط لتحويل المنظور"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """تحويل منظور الصورة"""
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

def detect_exam_contour(image):
    """كشف وتصحيح منظور ورقة الامتحان"""
    # تحسين الصورة
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # كشف الحواف
    edged = cv2.Canny(blurred, 50, 150)
    
    # العثور على الكنتورات
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # البحث عن أكبر كنتر رباعي (ورقة الامتحان)
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    
    if docCnt is not None:
        # تحويل منظور الصورة
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        return warped, True
    else:
        # إذا لم نجد ورقة، نستخدم الصورة كما هي
        return gray, False

def find_bubbles_debug(image):
    """كشف الفقاعات مع وضع التصحيح"""
    # استخدام adaptive threshold للتعامل مع الإضاءة المختلفة
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # تنظيف الصورة
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # العثور على الكنتورات
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    bubbles = []
    debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50 or area > 1000:  # نطاق مساحة الفقاعات
            continue
            
        # حساب الاستدارة
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:  # أقل استدارة مسموحة
            continue
            
        # الحصول على المستطيل المحيط
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        if 0.5 <= aspect_ratio <= 1.5:  # نطاق أوسع لنسبة الأبعاد
            # حساب نسبة التعبئة
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            filled_ratio = total / area
            
            bubbles.append({
                'contour': c,
                'center': (x + w//2, y + h//2),
                'bbox': (x, y, w, h),
                'filled_ratio': filled_ratio
            })
            
            # رسم على صورة التصحيح
            color = (0, 255, 0) if filled_ratio > 0.3 else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_img, f"{filled_ratio:.2f}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    print(f"🔍 تم اكتشاف {len(bubbles)} فقاعة")
    return bubbles, debug_img

def organize_and_extract_answers(bubbles, num_questions=10, options_per_question=4):
    """تنظيم الفقاعات واستخراج الإجابات"""
    if not bubbles:
        return ["فراغ"] * num_questions
    
    # ترتيب الفقاعات حسب الصفوف (من الأعلى إلى الأسفل)
    bubbles_sorted = sorted(bubbles, key=lambda b: b['center'][1])
    
    # تجميع الفقاعات في صفوف
    rows = []
    current_row = []
    row_height_threshold = 20  # مسافة أكبر بين الصفوف
    
    for bubble in bubbles_sorted:
        if not current_row:
            current_row.append(bubble)
        else:
            # التحقق إذا كانت الفقاعة في نفس الصف
            y_diff = abs(bubble['center'][1] - current_row[0]['center'][1])
            if y_diff <= row_height_threshold:
                current_row.append(bubble)
            else:
                # ترتيب الصف الحالي من اليسار لليمين
                current_row.sort(key=lambda b: b['center'][0])
                rows.append(current_row)
                current_row = [bubble]
    
    if current_row:
        current_row.sort(key=lambda b: b['center'][0])
        rows.append(current_row)
    
    print(f"📊 تم تجميع {len(rows)} صف")
    
    # استخراج الإجابات
    answers = []
    for i in range(min(num_questions, len(rows))):
        row = rows[i]
        if len(row) < options_per_question:
            answers.append("فراغ")
            continue
        
        # العثور على الفقاعة الأكثر تعبئة في الصف
        best_bubble = None
        best_ratio = 0
        
        for j, bubble in enumerate(row[:options_per_question]):
            if bubble['filled_ratio'] > best_ratio:
                best_ratio = bubble['filled_ratio']
                best_bubble = j
        
        # عتبة التعبئة (يمكن تعديلها)
        if best_ratio > 0.2:  # عتبة منخفضة للاختبار
            options = ['A', 'B', 'C', 'D', 'E'][:options_per_question]
            answers.append(options[best_bubble])
            print(f"✅ السؤال {i+1}: {options[best_bubble]} (نسبة التعبئة: {best_ratio:.2f})")
        else:
            answers.append("فراغ")
            print(f"❌ السؤال {i+1}: فراغ (أعلى نسبة: {best_ratio:.2f})")
    
    # تعبئة الأسئلة المتبقية
    while len(answers) < num_questions:
        answers.append("فراغ")
    
    return answers

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/correct', methods=['POST'])
def correct():
    """تصحيح ورقة الإجابة"""
    try:
        print("🚀 بدء عملية التصحيح...")
        
        # الحصول على المعلمات
        num_questions = request.form.get('num_questions', default=len(ANSWER_KEY), type=int)
        options_per_question = request.form.get('options_per_question', default=4, type=int)
        debug = request.form.get('debug', default=False, type=bool)
        
        # التحقق من وجود الملف
        if 'image' not in request.files:
            return jsonify({"error": "لم يتم تقديم صورة"}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "لم يتم اختيار ملف"}), 400
        
        print(f"📁 معالجة الصورة: {file.filename}")
        
        # قراءة الصورة
        nparr = np.frombuffer(file.read(), np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            return jsonify({"error": "فشل في قراءة الصورة"}), 400
        
        print(f"📐 أبعاد الصورة: {original_img.shape}")
        
        # كشف وتصحيح منظور الورقة
        processed_img, perspective_corrected = detect_exam_contour(original_img)
        print(f"🎯 تم تصحيح المنظور: {perspective_corrected}")
        
        # كشف الفقاعات
        bubbles, debug_img = find_bubbles_debug(processed_img)
        
        # استخراج الإجابات
        student_answers = organize_and_extract_answers(bubbles, num_questions, options_per_question)
        
        # حساب النتائج
        correct_count = sum(1 for s, c in zip(student_answers, ANSWER_KEY[:num_questions]) if s == c)
        wrong_count = num_questions - correct_count
        percentage = round((correct_count / num_questions) * 100) if num_questions > 0 else 0
        
        print(f"📊 النتائج: {correct_count} صحيحة، {wrong_count} خاطئة، {percentage}%")
        
        # تحضير الاستجابة
        response = {
            "success": True,
            "answers": student_answers,
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "percentage": percentage,
            "total_questions": num_questions,
            "bubbles_detected": len(bubbles),
            "perspective_corrected": perspective_corrected,
            "timestamp": datetime.now().isoformat()
        }
        
        # إضافة صورة التصحيح إذا طلب
        if debug and debug_img is not None:
            _, buffer = cv2.imencode('.jpg', debug_img)
            debug_base64 = base64.b64encode(buffer).decode('utf-8')
            response["debug_image"] = f"data:image/jpeg;base64,{debug_base64}"
            print("📷 تم إضافة صورة التصحيح")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ خطأ في التصحيح: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"خطأ داخلي في معالجة الصورة: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """فحص حالة الخادم"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "answer_key_count": len(ANSWER_KEY)
    })

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

# إصلاح مشكلة CORS
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
