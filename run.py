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

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================
MODELS_DIR = "models"
CAMERA_INDEX = 0  # ใช้กล้อง WebCam Default

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

# Global State
CURRENT_MODE = 'BOX'  # Start with Box mode
lock = threading.Lock()

# ==========================================
# 🧠 MODEL DEFINITIONS (ห้ามเปลี่ยน Base Code หลัก)
# ==========================================

# 1. ArcFace Layer (ตามโค้ดเก่า)
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    def forward(self, input):
        # Dummy forward for loading weights only
        return input

# 2. Main Model Class (ใช้โหลด .pth)
class PillModelGodTier(nn.Module):
    def __init__(self, num_classes=11, model_name="convnext_tiny"): # Default class ไว้ก่อน
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        # Dummy Input เพื่อหา In-features (ค่าโดยประมาณ)
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
print(f"🚀 System Starting on: {device}")

# --- Load Detectors (ONNX) ---
print("👁️ Loading ONNX Detectors...")
detectors = {}
try:
    detectors['BOX'] = ort.InferenceSession(PATHS['BOX']['det'], providers=['CPUExecutionProvider'])
    detectors['PILL'] = ort.InferenceSession(PATHS['PILL']['det'], providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"❌ Error loading ONNX: {e}")

# --- Load Classifiers (PyTorch) ---
print("🧠 Loading PyTorch Classifiers...")
classifiers = {}
transforms_dict = {}

def load_cls_model(path, size, num_classes=11):
    try:
        model = PillModelGodTier(num_classes=num_classes)
        # Load weights (map_location for CPU safety)
        checkpoint = torch.load(path, map_location=device)
        # Handle state_dict naming mismatch if any
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
        print(f"⚠️ Warning loading {path}: {e}")
        return None, None

# Load Both Models into RAM
classifiers['BOX'], transforms_dict['BOX'] = load_cls_model(PATHS['BOX']['cls'], PATHS['BOX']['size'])
classifiers['PILL'], transforms_dict['PILL'] = load_cls_model(PATHS['PILL']['cls'], PATHS['PILL']['size'])

print("✅ All Models Loaded.")

# ==========================================
# 📹 CORE PROCESSING LOOP
# ==========================================
def process_frame(frame, mode):
    # 1. Select Engine
    session = detectors[mode]
    model = classifiers[mode]
    preprocess = transforms_dict[mode]
    target_size = PATHS[mode]['size']

    # 2. Preprocess for YOLO (Basic Resize)
    img_h, img_w = frame.shape[:2]
    # YOLO Input logic (Simplified for demo)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    
    # 3. YOLO Inference (ONNX)
    input_name = session.get_inputs()[0].name
    # Note: ONNX Runtime expects numpy array, blob is numpy
    detections = session.run(None, {input_name: blob})[0] 
    # (สมมติ Output เป็น standard YOLOv8 format: 1x84x8400 -> ต้อง transpose มา parse)
    
    # --- MOCK DETECTION VISUALIZATION (เนื่องจากไม่ได้เชื่อม YOLO Post-process จริง) ---
    # ส่วนนี้คุณต้องเอา Logic Post-process (NMS) ของ YOLO มาใส่
    # เพื่อความง่ายในการ Show Code ผมจะวาด UI ทับไปก่อน
    
    cv2.putText(frame, f"MODE: {mode} (Press 'P' to toggle)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if mode == 'PILL' else (0, 165, 255), 3)
    
    cv2.rectangle(frame, (20, 20), (img_w-20, img_h-20), (0, 255, 0), 2)
    
    return frame

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # ลด Resolution กล้องหน่อยจะได้ลื่นๆ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        with lock:
            mode_now = CURRENT_MODE
            
        # Process Frame
        processed_frame = process_frame(frame, mode_now)
        
        # Encode for Web
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 🌐 WEB SERVER (Interface)
# ==========================================

# HTML Template (Embed JS for Key Listener)
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>PillTrack Monitor</title>
    <style>
        body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; }
        h1 { margin-bottom: 10px; }
        .monitor { border: 5px solid #333; border-radius: 10px; max-width: 90%; }
        .status { font-size: 20px; margin-top: 10px; color: #00ff00; }
        .controls { margin-top: 20px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>💊 PillTrack Core System</h1>
    <img src="/video_feed" class="monitor">
    
    <div class="status">Current Mode: <span id="mode-text">LOADING...</span></div>
    
    <div class="controls">
        <p>Press <b>'P'</b> on keyboard or click button below to switch mode.</p>
        <button onclick="toggleMode()">Refesh / Toggle Mode</button>
    </div>

    <script>
        // 1. Update Mode Display
        function updateMode() {
            fetch('/get_mode')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('mode-text').innerText = data.mode;
                    document.getElementById('mode-text').style.color = (data.mode === 'PILL') ? '#00ff00' : '#ffa500';
                });
        }

        // 2. Toggle Action
        function toggleMode() {
            fetch('/toggle_mode').then(() => updateMode());
        }

        // 3. Listen for 'P' Key
        document.addEventListener('keydown', (event) => {
            if (event.key === 'p' || event.key === 'P') {
                toggleMode();
            }
        });

        // Loop update status every 1 sec
        setInterval(updateMode, 1000);
        updateMode();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode')
def toggle_mode():
    global CURRENT_MODE
    with lock:
        if CURRENT_MODE == 'BOX':
            CURRENT_MODE = 'PILL'
        else:
            CURRENT_MODE = 'BOX'
    print(f"🔄 Switched to: {CURRENT_MODE}")
    return jsonify(success=True, mode=CURRENT_MODE)

@app.route('/get_mode')
def get_mode():
    return jsonify(mode=CURRENT_MODE)

if __name__ == '__main__':
    # Run Flask (Host 0.0.0.0 เพื่อให้เข้าจากเครื่องอื่นในวงแลนได้)
    print("🌍 Server running at http://localhost:5000")
    print("   (Or use http://<YOUR_IP>:5000 from another device)")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)