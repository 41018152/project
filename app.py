from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import easyocr
import re

app = Flask(__name__)

# 確認YOLO模型權重和配置文件的正確路徑
weights_path = "yolov3.weights"  # 確保這個文件存在
config_path = "yolov3.cfg"  # 確保這個文件存在

# 載入 YOLO 模型
net = cv2.dnn.readNet(weights_path, config_path)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 初始化 EasyOCR 讀取器
reader = easyocr.Reader(['en'])

def detect_license_plates(img):
    if img is None:
        raise ValueError("Image not loaded properly")
    
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    plates = []
    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plates.append(img[y:y+h, x:x+w])
    return img, plates

def extract_plate_number(plate_img):
    if plate_img is None or plate_img.size == 0:
        return ""
    
    # 優化影像處理
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 使用 EasyOCR 進行識別
    easyocr_text = reader.readtext(binary, detail=0)
    filtered_text = ''.join(re.findall(r'[A-Z0-9-]', ' '.join(easyocr_text).upper()))
    
    return filtered_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize_plate():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    npimg = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    processed_img, plates = detect_license_plates(img)
    
    if not plates:
        return jsonify({"message": "未檢測到車牌"})
    
    plate_numbers = [extract_plate_number(plate) for plate in plates]
    plate_number_str = ', '.join(plate_numbers)
    
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_str = base64.b64encode(buffer).decode()
    return jsonify({"image": img_str, "plateNumber": plate_number_str})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

