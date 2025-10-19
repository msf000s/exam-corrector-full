from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import imutils
import traceback

app = Flask(__name__, static_folder='static')
os.makedirs('static', exist_ok=True)

# ⚙️ عدّل هذا حسب اختبارك (10 أسئلة × 4 خيارات)
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

def extract_answers(warped_gray):
    # إحداثيات مخصصة لنموذجك (10 أسئلة × 4 خيارات)
    bubbles_coords = [
        # الجانب الأيمن (1-5)
          # السؤال 1 (الإحداثيات من خريطة الصورة)
    (670 - 16, 120 - 16, 32, 32), (612 - 17, 122 - 17, 34, 34), (552 - 14, 121 - 14, 28, 28),  (496 - 10, 117 - 10, 20, 20),  # D

        (920, 350, 40, 40), (870, 350, 40, 40), (820, 350, 40, 40), (770, 350, 40, 40),
        (920, 450, 40, 40), (870, 450, 40, 40), (820, 450, 40, 40), (770, 450, 40, 40),
        (920, 550, 40, 40), (870, 550, 40, 40), (820, 550, 40, 40), (770, 550, 40, 40),
        (920, 650, 40, 40), (870, 650, 40, 40), (820, 650, 40, 40), (770, 650, 40, 40),
        # الجانب الأيسر (6-10)
        (360, 250, 40, 40), (310, 250, 40, 40), (260, 250, 40, 40), (210, 250, 40, 40),
        (360, 350, 40, 40), (310, 350, 40, 40), (260, 350, 40, 40), (210, 350, 40, 40),
        (360, 450, 40, 40), (310, 450, 40, 40), (260, 450, 40, 40), (210, 450, 40, 40),
        (360, 550, 40, 40), (310, 550, 40, 40), (260, 550, 40, 40), (210, 550, 40, 40),
        (360, 650, 40, 40), (310, 650, 40, 40), (260, 650, 40, 40), (210, 650, 40, 40),
    ]
    options = ['A', 'B', 'C', 'D']
    answers = []
    for i in range(10):
        best_choice = -1
        max_filled = 0
        for c in range(4):
            idx = i * 4 + c
            x, y, w, h = bubbles_coords[idx]
            if y + h > warped_gray.shape[0] or x + w > warped_gray.shape[1]:
                answers.append("فراغ")
                continue
            roi = warped_gray[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            if total > max_filled:
                max_filled = total
                best_choice = c
        if max_filled < 50:
            answers.append("فراغ")
        else:
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

        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        student_answers = extract_answers(thresh)

        correct_count = sum(1 for s, c in zip(student_answers, ANSWER_KEY) if s == c)
        wrong_count = len(ANSWER_KEY) - correct_count
        percentage = round((correct_count / len(ANSWER_KEY)) * 100)
print("عدد الفقاعات المكتشفة:", len(bubbles_coords))
for i, (x, y, w, h) in enumerate(bubbles_coords):
    roi = warped_gray[y:y+h, x:x+w]
    total = cv2.countNonZero(roi)
    print(f"الفراغ {i}: مناطق مظللة = {total}")
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


