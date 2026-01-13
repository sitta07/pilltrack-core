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
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify
import threading
import logging
from collections import deque

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

# Optimized Configuration
CROP_EXPANSION_RATIO = 2.0
CONFIDENCE_THRESHOLD = 0.65
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_INDEX = 0

# Performance Optimizations
DETECTION_SKIP_FRAMES = 2  # Run detection every N frames
CLASSIFICATION_BATCH_SIZE = 1  # Keep at 1 for real-time
USE_HALF_PRECISION = torch.cuda.is_available()  # FP16 for GPU
FRAME_BUFFER_SIZE = 1  # Minimal buffering for low latency

# Global State
CURRENT_MODE = 'BOX'
ZOOM_LEVEL = 1.0
lock = threading.Lock()

# Frame counter for skipping
frame_counter = 0
last_detection_result = None
last_crop_coords = None

# ==========================================
# 🚀 OPTIMIZED MULTI-THREADING CAMERA CLASS
# ==========================================
class WebcamStream:
    """Optimized camera stream with reduced buffering"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # Optimized camera settings
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 🏗️ OPTIMIZED MODEL DEFINITION
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
    
    @torch.jit.export
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.bn_in(feat)
        feat = self.dropout(feat)
        emb = self.bn_emb(self.fc_emb(feat))
        return emb

# ==========================================
# 🛠️ OPTIMIZED HELPER FUNCTIONS
# ==========================================
def get_auto_hsv_bounds(frame, sample_size=20):
    """Reduced sample size for faster processing"""
    h, w, _ = frame.shape
    
    # Sample corners with smaller regions
    tl = frame[0:sample_size, 0:sample_size]
    tr = frame[0:sample_size, w-sample_size:w]
    bl = frame[h-sample_size:h, 0:sample_size]
    br = frame[h-sample_size:h, w-sample_size:w]
    
    samples = np.vstack((tl, tr, bl, br))
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv_samples, axis=(0, 1))
    
    lower_bound = mean - np.array([20, 50, 50])
    upper_bound = mean + np.array([20, 50, 50])
    
    lower_green = np.clip(lower_bound, [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    upper_green = np.clip(upper_bound, [0, 0, 0], [180, 255, 255]).astype(np.uint8)
    
    return lower_green, upper_green

def remove_green_bg_auto(image):
    """Optimized background removal"""
    if image is None or image.size == 0:
        return image
    
    lower, upper = get_auto_hsv_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    
    return result

# ==========================================
# 🚀 OPTIMIZED SYSTEM INIT & MODEL LOADING
# ==========================================
app = Flask(__name__)
loaded_systems = {}

def load_system(mode_name, config):
    """Optimized model loading with FP16 support"""
    print(f"⏳ Loading {mode_name} System...")
    sys_data = {}
    
    # 1. Load Detector with optimizations
    if not os.path.exists(config['det']):
        print(f"❌ {mode_name} DET not found: {config['det']}")
        return None
    
    sys_data['det_model'] = YOLO(config['det'], task='detect')
    
    # 2. Load Classification Model
    if not os.path.exists(config['cls']):
        print(f"❌ {mode_name} CLS not found: {config['cls']}")
        return None
    
    checkpoint = torch.load(config['cls'], map_location=DEVICE)
    class_names = checkpoint.get('class_names', ["Unknown"] * 100)
    img_size = checkpoint.get('img_size', config['img_size'])
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Auto-fix weight mismatch
    if 'head.weight' in state_dict:
        real_num_classes = state_dict['head.weight'].shape[0]
        if len(class_names) < real_num_classes:
            class_names.extend([f"Extra_{i}" for i in range(real_num_classes - len(class_names))])
        elif len(class_names) > real_num_classes:
            class_names = class_names[:real_num_classes]
    else:
        real_num_classes = len(class_names)
    
    model = PillModelGodTier(num_classes=real_num_classes, img_size=img_size).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Apply FP16 optimization for GPU
    if USE_HALF_PRECISION and torch.cuda.is_available():
        model = model.half()
        print(f"✅ {mode_name} using FP16 (Half Precision)")
    
    # Enable inference optimizations
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f"✅ {mode_name} compiled with torch.compile")
        except:
            pass
    
    sys_data['cls_model'] = model
    sys_data['class_names'] = class_names
    
    # Optimized transform pipeline
    sys_data['transform'] = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"✅ {mode_name} Loaded Successfully")
    return sys_data

# Load all systems
try:
    loaded_systems['BOX'] = load_system('BOX', MODELS['BOX'])
    loaded_systems['PILL'] = load_system('PILL', MODELS['PILL'])
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODELS: {e}")

# ==========================================
# 📹 OPTIMIZED CORE PROCESSING LOOP
# ==========================================
@torch.no_grad()  # Disable gradient computation for inference
def process_frame(frame):
    global ZOOM_LEVEL, CURRENT_MODE, frame_counter, last_detection_result, last_crop_coords
    
    system = loaded_systems.get(CURRENT_MODE)
    if not system:
        cv2.putText(frame, "Model Loading Error", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    # Digital Zoom (optimized)
    if ZOOM_LEVEL > 1.0:
        h, w = frame.shape[:2]
        new_w, new_h = int(w/ZOOM_LEVEL), int(h/ZOOM_LEVEL)
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h), 
                          interpolation=cv2.INTER_LINEAR)
    
    display_frame = frame.copy()
    h_main, w_main = display_frame.shape[:2]
    
    pill_crop_raw = None
    crop_coords = None
    
    # Frame skipping for detection (run every N frames)
    frame_counter += 1
    run_detection = (frame_counter % DETECTION_SKIP_FRAMES == 0)
    
    if run_detection:
        # Run YOLO detection
        results = system['det_model'](frame, verbose=False, imgsz=640, conf=0.5, half=USE_HALF_PRECISION)
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            max_idx = torch.argmax(boxes.conf).item()
            x1_raw, y1_raw, x2_raw, y2_raw = boxes.xyxy[max_idx].cpu().numpy()
            
            # Expanded crop logic
            w_box, h_box = x2_raw - x1_raw, y2_raw - y1_raw
            cx, cy = x1_raw + w_box/2, y1_raw + h_box/2
            
            side = max(w_box, h_box) * CROP_EXPANSION_RATIO
            
            x1 = int(max(0, cx - side//2))
            y1 = int(max(0, cy - side//2))
            x2 = int(min(w_main, cx + side//2))
            y2 = int(min(h_main, cy + side//2))
            
            crop_coords = (x1, y1, x2-x1, y2-y1)
            last_detection_result = boxes
            last_crop_coords = crop_coords
        else:
            last_detection_result = None
            last_crop_coords = None
    else:
        # Reuse last detection
        crop_coords = last_crop_coords
    
    # Extract crop if available
    if crop_coords is not None:
        x1, y1, w, h = crop_coords
        pill_crop_raw = frame[y1:y1+h, x1:x1+w]
    
    # Classification
    final_input_img = None
    class_name = "Scanning..."
    conf_val = 0.0
    color = (100, 100, 100)
    
    if pill_crop_raw is not None and pill_crop_raw.size > 0:
        final_input_img = remove_green_bg_auto(pill_crop_raw)
        
        # Convert and prepare input
        img_rgb = cv2.cvtColor(final_input_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        input_tensor = system['transform'](img_pil).unsqueeze(0).to(DEVICE)
        
        # Apply FP16 if enabled
        if USE_HALF_PRECISION and torch.cuda.is_available():
            input_tensor = input_tensor.half()
        
        # Inference
        emb = system['cls_model'](input_tensor)
        logits = F.linear(F.normalize(emb), F.normalize(system['cls_model'].head.weight))
        probs = F.softmax(logits * 30.0, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        conf_val = confidence.item()
        idx_val = predicted_idx.item()
        
        if conf_val > CONFIDENCE_THRESHOLD:
            class_name = system['class_names'][idx_val] if idx_val < len(system['class_names']) else f"Class_{idx_val}"
            color = (0, 255, 0)
        else:
            class_name = "Unknown"
            color = (0, 0, 255)
        
        # Draw bounding box
        if crop_coords:
            cx, cy, cw, ch = crop_coords
            cv2.rectangle(display_frame, (cx, cy), (cx+cw, cy+ch), color, 3)
    
    # UI Overlay - Optimized drawing
    if final_input_img is not None:
        try:
            preview_size = 150
            display_crop = cv2.resize(final_input_img, (preview_size, preview_size), 
                                     interpolation=cv2.INTER_LINEAR)
            display_crop = cv2.copyMakeBorder(display_crop, 2, 2, 2, 2, 
                                             cv2.BORDER_CONSTANT, value=color)
            
            h_crop, w_crop = display_crop.shape[:2]
            x_pos = w_main - w_crop - 20
            y_pos = 100
            
            if y_pos + h_crop < h_main and x_pos + w_crop < w_main:
                display_frame[y_pos:y_pos+h_crop, x_pos:x_pos+w_crop] = display_crop
        except:
            pass
    
    # Status bar
    cv2.rectangle(display_frame, (0, 0), (w_main, 40), (0, 0, 0), -1)
    cv2.putText(display_frame, f"MODE: {CURRENT_MODE} (Press 'P')", 
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Info box
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 50), (350, 150), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
    
    cv2.putText(display_frame, f"CLASS: {class_name}", 
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"CONF:  {conf_val*100:.1f}%", 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return display_frame

def generate_frames():
    """Optimized frame generation with better threading"""
    vs = WebcamStream(src=CAMERA_INDEX).start()
    time.sleep(1.0)  # Reduced warmup time
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
        
        with lock:
            processed = process_frame(frame)
        
        # Optimized JPEG encoding
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # Slightly lower quality for speed
        ret, buffer = cv2.imencode('.jpg', processed, encode_param)
        
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    vs.stop()

# ==========================================
# 🌐 WEB SERVER
# ==========================================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>PillTrack System - Optimized</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
    body { 
        background: #111; 
        color: white; 
        text-align: center; 
        font-family: 'Segoe UI', sans-serif; 
        margin: 0;
        padding: 0;
        overflow: hidden; 
    }
    img { 
        max-width: 100%; 
        height: auto; 
        border: 2px solid #333;
        display: block;
        margin: 0 auto;
    }
    .status { 
        margin-top: 10px; 
        font-size: 24px; 
        font-weight: bold; 
    }
    .info {
        font-size: 14px;
        color: #888;
        margin-top: 5px;
    }
</style>
</head>
<body>
    <img src="/video_feed" id="videoFeed">
    <div class="status">
        Mode: <span id="mode" style="color:yellow">LOADING</span>
    </div>
    <div class="info">Press 'P' to toggle mode | Optimized for Speed</div>
    
    <script>
        function updateMode() {
            fetch('/get_mode')
                .then(r => r.json())
                .then(d => {
                    const el = document.getElementById('mode');
                    el.innerText = d.mode;
                    el.style.color = d.mode === 'PILL' ? '#0f0' : '#fa0';
                })
                .catch(err => console.error('Mode update failed:', err));
        }
        
        function toggleMode() { 
            fetch('/toggle_mode').then(updateMode); 
        }
        
        document.addEventListener('keydown', e => {
            if(e.key.toLowerCase() === 'p') toggleMode();
        });
        
        // Update mode every 2 seconds (reduced frequency)
        setInterval(updateMode, 2000);
        updateMode();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_mode', methods=['GET', 'POST'])
def toggle_mode():
    global CURRENT_MODE, frame_counter, last_detection_result, last_crop_coords
    with lock:
        CURRENT_MODE = 'PILL' if CURRENT_MODE == 'BOX' else 'BOX'
        # Reset detection cache on mode change
        frame_counter = 0
        last_detection_result = None
        last_crop_coords = None
    return jsonify(success=True, mode=CURRENT_MODE)

@app.route('/get_mode')
def get_mode():
    return jsonify(mode=CURRENT_MODE)

# ==========================================
# 🚀 MAIN ENTRY POINT
# ==========================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("🚀 OPTIMIZED PILLTRACK SYSTEM")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Half Precision: {USE_HALF_PRECISION}")
    print(f"Detection Skip Frames: {DETECTION_SKIP_FRAMES}")
    print(f"Server: http://0.0.0.0:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)