import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time
import os
import psutil
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify
import threading
import logging

# ==========================================
# ⚙️ CONFIGURATION & PATHS
# ==========================================
MODELS = {
    'BOX': {
        'cls': 'models/box_model_production.pth',
        'det': 'models/box_detector.onnx',
        'img_size': 640
    },
    'PILL': {
        'cls': 'models/pill_model_production.pth',
        'det': 'models/pill_detector.onnx',
        'img_size': 384
    }
}

CONFIDENCE_THRESHOLD = 0.65
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_INDEX = 0

# Global State
CURRENT_MODE = 'BOX'
ZOOM_LEVEL = 1.0
ZOOM_STEP = 0.1
lock = threading.Lock()

# ==========================================
# 🏗️ MODEL DEFINITION
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label=None):
        return F.linear(F.normalize(input), F.normalize(self.weight)) * self.s

class PillModelGodTier(nn.Module):
    def __init__(self, num_classes, model_name="convnext_tiny", img_size=640):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
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
# 🛠️ HELPER FUNCTIONS
# ==========================================
def get_auto_hsv_bounds(frame, sample_size=30):
    h, w, _ = frame.shape
    tl = frame[0:sample_size, 0:sample_size]
    tr = frame[0:sample_size, w-sample_size:w]
    bl = frame[h-sample_size:h, 0:sample_size]
    br = frame[h-sample_size:h, w-sample_size:w]
    samples = np.vstack((tl, tr, bl, br))
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv_samples, axis=(0, 1))
    
    lower_bound = mean - np.array([20, 50, 50])
    upper_bound = mean + np.array([20, 50, 50])
    
    lower_green = np.clip(lower_bound, np.array([0, 0, 0]), np.array([180, 255, 255])).astype(np.uint8)
    upper_green = np.clip(upper_bound, np.array([0, 0, 0]), np.array([180, 255, 255])).astype(np.uint8)
    return lower_green, upper_green

def remove_green_bg_auto(image):
    if image is None or image.size == 0: return image
    lower, upper = get_auto_hsv_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

class PerformanceMonitor:
    def __init__(self):
        self.seg_time = 0; self.cls_time = 0; self.total_time = 0
        self.cpu_usage = 0; self.ram_usage = 0; self.last_update = time.time()
    def update_system_stats(self):
        if time.time() - self.last_update > 1.0:
            self.cpu_usage = psutil.cpu_percent()
            self.ram_usage = psutil.virtual_memory().percent
            self.last_update = time.time()
    def get_overlay_text(self):
        return [f"CPU: {self.cpu_usage}% | RAM: {self.ram_usage}%",
                f"DET: {self.seg_time*1000:.1f}ms | CLS: {self.cls_time*1000:.1f}ms"]

# ==========================================
# 🚀 SYSTEM INIT & MODEL LOADING (FIXED)
# ==========================================
app = Flask(__name__)
loaded_systems = {}

def load_system(mode_name, config):
    print(f"⏳ Loading {mode_name} System...")
    sys_data = {}
    
    # 1. Load Detector (.onnx)
    if not os.path.exists(config['det']):
        print(f"❌ {mode_name} DET not found: {config['det']}")
        return None
    sys_data['det_model'] = YOLO(config['det'], task='detect')

    # 2. Load CLS Model (.pth) - 🔧 FIX: Trust Weight Shape
    if not os.path.exists(config['cls']):
        print(f"❌ {mode_name} CLS not found: {config['cls']}")
        return None
        
    checkpoint = torch.load(config['cls'], map_location=DEVICE)
    
    # Extract Metadata
    class_names = checkpoint.get('class_names', ["Unknown"] * 100)
    img_size = checkpoint.get('img_size', config['img_size'])
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 🔧 FIX: Check actual weight shape from state_dict
    if 'head.weight' in state_dict:
        real_num_classes = state_dict['head.weight'].shape[0]
        print(f"   🔧 Auto-fix: Weights have {real_num_classes} classes (Metadata said {len(class_names)})")
        
        # Adjust class_names list to match weights (Prevent Crash)
        if len(class_names) < real_num_classes:
            diff = real_num_classes - len(class_names)
            class_names.extend([f"Extra_{i}" for i in range(diff)])
        elif len(class_names) > real_num_classes:
            class_names = class_names[:real_num_classes]
    else:
        real_num_classes = len(class_names)

    # Init Model with REAL shape
    model = PillModelGodTier(num_classes=real_num_classes, img_size=img_size).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    sys_data['cls_model'] = model
    sys_data['class_names'] = class_names
    sys_data['transform'] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"✅ {mode_name} Loaded.")
    return sys_data

# Load Everything
try:
    loaded_systems['BOX'] = load_system('BOX', MODELS['BOX'])
    loaded_systems['PILL'] = load_system('PILL', MODELS['PILL'])
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODELS: {e}")

monitor = PerformanceMonitor()

# ==========================================
# 📹 CORE PROCESSING LOOP
# ==========================================
def process_frame(frame):
    global ZOOM_LEVEL, CURRENT_MODE
    loop_start = time.time()
    monitor.update_system_stats()
    
    system = loaded_systems.get(CURRENT_MODE)
    if not system:
        cv2.putText(frame, "Model Loading Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # Digital Zoom
    if ZOOM_LEVEL > 1.0:
        h, w, _ = frame.shape
        new_w, new_h = int(w/ZOOM_LEVEL), int(h/ZOOM_LEVEL)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

    display_frame = frame.copy()
    h_main, w_main, _ = display_frame.shape
    
    pill_crop_raw = None
    crop_coords = None

    # 1. DETECT
    t0 = time.time()
    results = system['det_model'](frame, verbose=False, imgsz=640, conf=0.6)
    monitor.seg_time = time.time() - t0

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        max_idx = torch.argmax(boxes.conf).item()
        x1_raw, y1_raw, x2_raw, y2_raw = boxes.xyxy[max_idx].cpu().numpy()
        
        w_box, h_box = x2_raw - x1_raw, y2_raw - y1_raw
        cx, cy = x1_raw + w_box/2, y1_raw + h_box/2
        side = max(w_box, h_box) + 40
        
        x1 = int(max(0, cx - side//2))
        y1 = int(max(0, cy - side//2))
        x2 = int(min(w_main, cx + side//2))
        y2 = int(min(h_main, cy + side//2))
        
        pill_crop_raw = frame[y1:y2, x1:x2]
        crop_coords = (x1, y1, x2-x1, y2-y1)

    # 2. CLASSIFY
    final_input_img = None 
    class_name = "Scanning..."
    conf_val = 0.0
    color = (100, 100, 100)

    if pill_crop_raw is not None and pill_crop_raw.size > 0:
        final_input_img = remove_green_bg_auto(pill_crop_raw)

        t1 = time.time()
        img_rgb = cv2.cvtColor(final_input_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        input_tensor = system['transform'](img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            emb = system['cls_model'](input_tensor)
            logits = F.linear(F.normalize(emb), F.normalize(system['cls_model'].head.weight))
            probs = F.softmax(logits * 30.0, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        monitor.cls_time = time.time() - t1
        
        conf_val = confidence.item()
        idx_val = predicted_idx.item()
        
        if conf_val > CONFIDENCE_THRESHOLD:
            # Safety check for index out of range
            if idx_val < len(system['class_names']):
                class_name = system['class_names'][idx_val]
            else:
                class_name = f"Class_{idx_val}"
            color = (0, 255, 0)
        else:
            class_name = "Unknown"
            color = (0, 0, 255)

        cx, cy, cw, ch = crop_coords
        cv2.rectangle(display_frame, (cx, cy), (cx+cw, cy+ch), color, 3)

    # 3. UI
    if final_input_img is not None:
        try:
            preview_size = 150
            display_crop = cv2.resize(final_input_img, (preview_size, preview_size))
            display_crop = cv2.copyMakeBorder(display_crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
            h_crop, w_crop, _ = display_crop.shape
            x_pos = w_main - w_crop - 20
            y_pos = 100
            if y_pos + h_crop < h_main and x_pos + w_crop < w_main:
                display_frame[y_pos:y_pos+h_crop, x_pos:x_pos+w_crop] = display_crop
        except: pass

    monitor.total_time = time.time() - loop_start
    
    # HUD
    cv2.rectangle(display_frame, (0, 0), (w_main, 40), (0,0,0), -1)
    cv2.putText(display_frame, f"MODE: {CURRENT_MODE} (Press 'P')", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 50), (350, 200), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
    
    cv2.putText(display_frame, f"CLASS: {class_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"CONF:  {conf_val*100:.1f}%", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    stats = monitor.get_overlay_text()
    for i, line in enumerate(stats):
        cv2.putText(display_frame, line, (10, 150 + (i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return display_frame

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ Camera Error")
        return
    while True:
        success, frame = cap.read()
        if not success: break
        with lock:
            processed = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 🌐 WEB SERVER
# ==========================================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>PillTrack System</title>
<style>
    body { background: #111; color: white; text-align: center; font-family: sans-serif; overflow: hidden; }
    img { max-width: 100%; height: auto; border: 2px solid #333; }
    .status { margin-top: 10px; font-size: 24px; font-weight: bold; }
</style>
</head>
<body>
    <img src="/video_feed">
    <div class="status">Current Mode: <span id="mode" style="color:yellow">LOADING</span></div>
    <script>
        function updateMode() {
            fetch('/get_mode').then(r=>r.json()).then(d=>{
                const el = document.getElementById('mode');
                el.innerText = d.mode;
                el.style.color = d.mode === 'PILL' ? '#0f0' : '#fa0';
            });
        }
        function toggle() { fetch('/toggle_mode').then(updateMode); }
        document.addEventListener('keydown', e => {
            if(e.key.toLowerCase() === 'p') toggle();
        });
        setInterval(updateMode, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode')
def toggle_mode():
    global CURRENT_MODE
    with lock:
        CURRENT_MODE = 'PILL' if CURRENT_MODE == 'BOX' else 'BOX'
    return jsonify(success=True, mode=CURRENT_MODE)

@app.route('/get_mode')
def get_mode(): return jsonify(mode=CURRENT_MODE)

if __name__ == '__main__':
    print("🌍 Server running: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)