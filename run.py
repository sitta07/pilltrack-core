import cv2
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import timm
from PIL import Image
from torchvision import transforms
import time
from flask import Flask, Response, render_template_string, jsonify
import threading
import os
import logging

# ==========================================
# 🔧 DEBUG & LOGGING CONFIG
# ==========================================
# ตั้งค่า Log ให้โชว์เวลาและระดับความรุนแรง
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # โชว์ใน Terminal
        logging.FileHandler("system_debug.log")  # บันทึกลงไฟล์เผื่อกลับมาดู
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================
MODELS_DIR = "models"
CAMERA_INDEX = 0

# Thresholds (ปรับตรงนี้ถ้า detect ยาก/ง่ายไป)
CONF_THRESHOLD = 0.4  # ความมั่นใจขั้นต่ำ (0.0 - 1.0)
IOU_THRESHOLD = 0.5   # การซ้อนทับของกล่อง

PATHS = {
    'BOX': {
        'det': os.path.join(MODELS_DIR, "box_detector.onnx"),
        'cls': os.path.join(MODELS_DIR, "box_model_production.pth"),
        'size': 640
    },
    'PILL': {
        'det': os.path.join(MODELS_DIR, "pill_detector.onnx"),
        'cls': os.path.join(MODELS_DIR, "pill_model_production.pth"),
        'size': 384
    }
}

CURRENT_MODE = 'BOX'
lock = threading.Lock()

# ==========================================
# 🧠 MODEL DEFINITIONS (Base Code)
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    def forward(self, input):
        return input

class PillModelGodTier(nn.Module):
    def __init__(self, num_classes=11, model_name="convnext_tiny"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        dummy = torch.randn(1, 3, 224, 224)
        in_features = self.backbone(dummy).shape[1]
        self.bn_in = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_emb = nn.Linear(in_features, 512)
        self.bn_emb = nn.BatchNorm1d(512)
        self.head = ArcMarginProduct(in_features=512, out_features=num_classes)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.bn_in(feat)
        feat = self.dropout(feat)
        emb = self.bn_emb(self.fc_emb(feat))
        return emb

# ==========================================
# 🛠️ SYSTEM INITIALIZATION
# ==========================================
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 System Starting on Device: {device}")

# --- Load Detectors ---
logger.info("👁️ Loading ONNX Detectors...")
detectors = {}
try:
    # Check files exist
    if not os.path.exists(PATHS['BOX']['det']):
        logger.error(f"❌ File not found: {PATHS['BOX']['det']}")
    
    # Load Models
    detectors['BOX'] = ort.InferenceSession(PATHS['BOX']['det'], providers=['CPUExecutionProvider'])
    detectors['PILL'] = ort.InferenceSession(PATHS['PILL']['det'], providers=['CPUExecutionProvider'])
    logger.info("✅ Detectors Loaded Successfully")
except Exception as e:
    logger.critical(f"❌ Error loading ONNX: {e}")

# --- Load Classifiers ---
logger.info("🧠 Loading Classifiers...")
classifiers = {}
transforms_dict = {}

def load_cls_model(path, size, num_classes=11):
    try:
        if not os.path.exists(path):
            logger.warning(f"⚠️ Model file missing: {path}")
            return None, None
            
        model = PillModelGodTier(num_classes=num_classes)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return model, preprocess
    except Exception as e:
        logger.error(f"❌ Error loading Classifier {path}: {e}")
        return None, None

classifiers['BOX'], transforms_dict['BOX'] = load_cls_model(PATHS['BOX']['cls'], PATHS['BOX']['size'])
classifiers['PILL'], transforms_dict['PILL'] = load_cls_model(PATHS['PILL']['cls'], PATHS['PILL']['size'])

# ==========================================
# 🕵️ YOLO UTILS (The Missing Piece)
# ==========================================
def parse_yolo_output(outputs, img_w, img_h):
    # YOLOv8 Output shape: (1, 4 + num_classes, 8400) -> e.g. (1, 5, 8400)
    # 4 coords (cx, cy, w, h) + probability
    
    output = outputs[0]  # (1, 84, 8400)
    output = output.transpose() # (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    # Loop through all 8400 rows
    for row in output:
        # Get max confidence score
        classes_scores = row[4:]
        if len(classes_scores) == 0: continue # Safety
        
        _, max_score, _, max_class_loc = cv2.minMaxLoc(classes_scores)
        
        if max_score > CONF_THRESHOLD:
            # Extract Box
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # Convert to Top-Left corner
            x1 = int((cx - w/2) * img_w / 640) # Scale back to original image
            y1 = int((cy - h/2) * img_h / 640)
            w_px = int(w * img_w / 640)
            h_px = int(h * img_h / 640)
            
            boxes.append([x1, y1, w_px, h_px])
            confidences.append(float(max_score))
            class_ids.append(max_class_loc[1])

    # NMS (Non-Maximum Suppression) to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((boxes[i], confidences[i], class_ids[i]))
            
    return results

# ==========================================
# 📹 CORE PROCESSING LOOP
# ==========================================
def process_frame(frame, mode):
    start_time = time.time()
    img_h, img_w = frame.shape[:2]
    
    # 1. Select Engine
    session = detectors.get(mode)
    if session is None:
        cv2.putText(frame, "Model Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # 2. Preprocess for YOLO
    # YOLOv8 expects RGB, Normalized 0-1, (1, 3, 640, 640)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    
    # 3. Inference
    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: blob})
        
        # 4. Post-process (Parse Output)
        detections = parse_yolo_output(outputs, img_w, img_h)
        
        # Log detections count
        if len(detections) > 0:
            logger.debug(f"📸 Mode {mode}: Found {len(detections)} objects")

        # 5. Draw & Classify
        for (box, score, cls_id) in detections:
            x, y, w, h = box
            
            # --- Draw Box ---
            color = (0, 255, 0) if mode == 'PILL' else (255, 165, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # --- Prepare text ---
            label = f"Conf: {score:.2f}"
            
            # Optional: Add Classification Logic Here
            # crop = frame[y:y+h, x:x+w]
            # ... pass crop to classifiers[mode] ...
            
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    except Exception as e:
        logger.error(f"Inference Error: {e}")

    # 6. FPS & Status
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"MODE: {mode} | FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

def generate_frames():
    logger.info(f"📷 Opening Camera Index: {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        logger.critical("❌ Could not open webcam! Check USB connection.")
        return

    # Optimization
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("⚠️ Failed to read frame from camera.")
            time.sleep(1)
            continue
            
        frame_count += 1
        
        with lock:
            mode_now = CURRENT_MODE
            
        # Process every frame
        processed_frame = process_frame(frame, mode_now)
        
        # Log ping every 100 frames to ensure alive
        if frame_count % 100 == 0:
            logger.info("🟢 System Alive - Streaming...")

        try:
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"Encoding error: {e}")

# ==========================================
# 🌐 WEB SERVER (Interface)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>PillTrack Monitor</title>
    <style>
        body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; }
        .monitor { border: 2px solid #555; max-width: 90%; }
        button { padding: 15px 30px; font-size: 18px; margin-top: 20px; 
                 background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px;}
        button:hover { background: #0056b3; }
        .log-box { margin-top: 20px; font-family: monospace; color: #aaa; font-size: 12px; }
    </style>
</head>
<body>
    <h1>💊 PillTrack Debug Console</h1>
    <img src="/video_feed" class="monitor">
    <h2 id="mode-text">Current Mode: LOADING...</h2>
    <button onclick="toggleMode()">Click or Press 'P' to Switch Mode</button>
    <div class="log-box">Check terminal for detailed logs...</div>

    <script>
        function updateMode() {
            fetch('/get_mode')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('mode-text').innerText = "Mode: " + d.mode;
                    document.getElementById('mode-text').style.color = (d.mode === 'PILL') ? '#00ff00' : '#ffa500';
                });
        }
        function toggleMode() { fetch('/toggle_mode').then(() => updateMode()); }
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'p' || e.key === 'P') toggleMode();
        });
        
        setInterval(updateMode, 1000);
        updateMode();
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode')
def toggle_mode():
    global CURRENT_MODE
    with lock:
        CURRENT_MODE = 'PILL' if CURRENT_MODE == 'BOX' else 'BOX'
    logger.info(f"🔄 Mode Switched to: {CURRENT_MODE}")
    return jsonify(success=True, mode=CURRENT_MODE)

@app.route('/get_mode')
def get_mode(): return jsonify(mode=CURRENT_MODE)

if __name__ == '__main__':
    logger.info("🌍 Server starting at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)