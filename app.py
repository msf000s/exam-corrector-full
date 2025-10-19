from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import imutils
import traceback

app = Flask(__name__, static_folder='static')
os.makedirs('static', exist_ok=True)

# ⚙️ الإجابات الصحيحة (عدّلها حسب اختبارك)
ANSWER_KEY = ['B', 'C', 'A', 'D', 'B', 'A', 'C', 'D', 'A', 'B']

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
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def extract_answers(warped_gray, num_questions=10, options_per_q=4):
    """
    اكتشف الفقاعات تلقائيًا وحدد الإجابة المظللة في كل سؤال
    """
    # 1. كشف الحواف
    edges = cv2.Canny(warped_gray, 50, 150)

    # 2. العثور على الكنتورات (الحدود)
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    bubbles = []
    for c in cnts:
        # حساب محيط الكنتور
        peri = cv2.arcLength(c, True)
        # تقريب الشكل
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # نختار فقط الكنتورات التي تشبه الدوائر (4+ نقاط)
        if len(approx) >= 5:
            # حساب مركز الكنتور
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # حساب نصف القطر التقريبي
                _, _, w, h = cv2.boundingRect(c)
                radius = min(w, h) // 2

                # تحقق من أن الحجم مناسب للفقاعة (10-50 بكسل)
                if 10 < radius < 50:
                    bubbles.append((cX, cY, radius))

    # 3. ترتيب الفقاعات حسب الموضع (من الأعلى لليسار)
    bubbles = sorted(bubbles, key=lambda b: (b[1] // 50, b[0]))

    # 4. تجميع الفقاعات حسب الأسئلة
    answers = []
    for i in range(num_questions):
        question_bubbles = bubbles[i * options_per_q:(i + 1) * options_per_q]
        if len(question_bubbles) != options_per_q:
            answers.append("فراغ")
            continue

        best_choice = -1
        max_filled = 0

        for j, (x, y, r) in enumerate(question_bubbles):
            # استخراج منطقة الفقاعة
            roi_x = max(0, x - r)
            roi_y = max(0, y - r)
            roi_w = min(warped_gray.shape[1] - roi_x, 2 * r)
            roi_h = min(warped_gray.shape[0] - roi_y, 2 * r)

            roi = warped_gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            total = cv2.countNonZero(roi)

            if total > max_filled:
                max_filled = total
                best_choice = j

        # عتبة لتحديد "فارغ" (يمكنك تعديلها)
        if max_filled < 50:
            answers.append("فراغ")
        else:
            options = ['A', 'B', 'C', 'D', 'E'][:options_per_q]
            answers.append(options[best_choice])

    return answers

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/correct', methods=['POST'])
def correct():
    try:
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "فشل في فك تشفير الصورة"}), 400

        orig = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

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

        if docCnt is not None:
            warped = four_point_transform(gray, docCnt.reshape(4, 2))
        else:
            warped = gray

        # افترض أنك تعرف عدد الأسئلة من واجهتك (يمكن إرساله في الطلب)
        num_questions = len(ANSWER_KEY)  # مثلاً 10
        options_per_q = 4  # مثلاً 4 خيارات

        student_answers = extract_answers(warped, num_questions, options_per_q)

        correct_count = sum(1 for s, c in zip(student_answers, ANSWER_KEY) if s == c)
        wrong_count = len(ANSWER_KEY) - correct_count
        percentage = round((correct_count / len(ANSWER_KEY)) * 100)

        return jsonify({
            "answers": student_answers,
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "percentage": percentage
        })

    except Exception as e:
        print("❌ خطأ في /correct:", str(e))
        traceback.print_exc()
        return jsonify({"error": "خطأ داخلي في معالجة الصورة"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
