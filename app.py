import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import imutils

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ⚙️ عدّل هذا حسب اختبارك
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}  # 5 أسئلة: B, E, A, D, B

# --- دوال معالجة الصور ---
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct():
    if 'image' not in request.files:
        return jsonify({"error": "لم يتم تحميل صورة"}), 400

    file = request.files['image']
    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    file.save(input_path)

    image = cv2.imread(input_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        paper = four_point_transform(orig, docCnt.reshape(4, 2))
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
    else:
        paper = orig
        warped = gray

    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            questionCnts.append(c)

    if len(questionCnts) < len(ANSWER_KEY) * 5:
        return jsonify({"error": "لم يتم اكتشاف عدد كافٍ من الفقاعات"}), 400

    questionCnts = sorted(questionCnts, key=lambda ctr: (cv2.boundingRect(ctr)[1] // 30, cv2.boundingRect(ctr)[0]))
    correct = 0

    for (q, i) in enumerate(range(0, len(questionCnts), 5)):
        if q >= len(ANSWER_KEY):
            break
        cnts = sorted(questionCnts[i:i+5], key=lambda ctr: cv2.boundingRect(ctr)[0])
        bubbled = None
        max_val = 0
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total > max_val:
                max_val = total
                bubbled = j
        if bubbled == ANSWER_KEY[q]:
            correct += 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.drawContours(paper, [cnts[bubbled]], -1, color, 3)

    score = (correct / len(ANSWER_KEY)) * 100
    output_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(output_path, paper)

    return jsonify({
        "score": round(score, 2),
        "correct": correct,
        "total": len(ANSWER_KEY),
        "result_image": "/static/result.jpg"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)