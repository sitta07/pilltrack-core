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

PADDING_RATIO = 0.10 
CONFIDENCE_THRESHOLD = 0.65
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global State
CURRENT_MODE = 'BOX'
ZOOM_LEVEL = 1.0
lock = threading.Lock()

# ==========================================
# 🛠️ AUTO-DETECT CAMERA FUNCTION (KEY FIX 🔑)
# ==========================================
def find_working_camera():
    """วนหา Camera Index ที่ใช้งานได้จริง (0-5)"""
    print("🔍 Searching for available camera...")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                print(f"✅ Found working camera at Index: {index}")
                return index
            cap.release()
    print("❌ No physical camera found. Using dummy mode.")
    return None

# หา Index จริงก่อนเริ่มระบบ
CAMERA_INDEX = find_working_camera()

# ==========================================
# 🚀 WEBCAM STREAM (ROBUST VERSION)
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        if src is None:
            self.stream = None
            self.grabbed = False
            self.frame = self.create_dummy_frame("NO CAMERA FOUND")
        else:
            self.stream = cv2.VideoCapture(src)
            # พยายามปรับความละเอียด (อาจไม่ได้ผลกับบางกล้อง แต่ใส่ไว้ดีกว่า)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            if self.stream and self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                if grabbed and frame is not None:
                    with lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    # ถ้าอ่านไม่ได้ ให้ใช้ Dummy Frame แทนการ Crash
                    self.grabbed = False
                    with lock:
                        self.frame = self.create_dummy_frame("CAMERA DISCONNECTED")
            else:
                # กรณีไม่มี Stream ตั้งแต่ต้น
                time.sleep(0.1)

    def read(self):
        with lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()

    def create_dummy_frame(self, text):
        """สร้างภาพสีดำพร้อมข้อความแจ้งเตือน"""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, text, (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0, 0, 255), 3, cv2.LINE_AA)
        return img

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
        # ใช้ try-catch ป้องกัน timm download error
        try:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        except:
            self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0) # Fallback

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
    if frame is None or frame.shape[0] < sample_size or frame.shape[1] < sample_size:
        return np.array([0,0,0]), np.array([180,255,255])
        
    h, w, _ = frame.shape
    tl = frame[0:sample_size, 0:sample_size]
    tr = frame[0:sample_size, w-sample_size:w]
    bl = frame[h-sample_size:h, 0:sample_size]
    br = frame[h-sample_size:h, w-sample_size:w]
    samples = np.vstack((tl, tr, bl, br))
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv_samples, axis=(0, 1))
    lower_green = np.clip(mean - np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    upper_green = np.clip(mean + np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    return lower_green, upper_green

def remove_green_bg_auto(image):
    if image is None or image.size == 0: return image
    lower, upper = get_auto_hsv_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

# ==========================================
# 🚀 SYSTEM INIT
# ==========================================
app = Flask(__name__)
loaded_systems = {}

def load_system(mode_name, config):
    print(f"⏳ Loading {mode_name} System...")
    sys_data = {}
    
    # 1. Load DETECTOR
    if os.path.exists(config['det']):
        sys_data['det_model'] = YOLO(config['det'], task='segment') 
    else:
        print(f"⚠️ Warning: {config['det']} not found. Skipping detector.")
        sys_data['det_model'] = None

    # 2. Load CLASSIFIER
    if os.path.exists(config['cls']):
        checkpoint = torch.load(config['cls'], map_location=DEVICE)
        class_names = checkpoint.get('class_names', ["Unknown"] * 100)
        img_size = checkpoint.get('img_size', config['img_size'])
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        if 'head.weight' in state_dict:
            real_num_classes = state_dict['head.weight'].shape[0]
            if len(class_names) < real_num_classes:
                class_names.extend([f"Extra_{i}" for i in range(real_num_classes - len(class_names))])
            elif len(class_names) > real_num_classes:
                class_names = class_names[:real_num_classes]
        else:
            real_num_classes = len(class_names)

        model = PillModelGodTier(num_classes=real_num_classes, img_size=img_size).to(DEVICE)
        try:
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            sys_data['cls_model'] = model
            sys_data['class_names'] = class_names
            sys_data['transform'] = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"❌ Error loading classifier weights: {e}")
            return None
    else:
        print(f"⚠️ Warning: {config['cls']} not found.")
        return None
    
    print(f"✅ {mode_name} Loaded.")
    return sys_data

try:
    loaded_systems['BOX'] = load_system('BOX', MODELS['BOX'])
    loaded_systems['PILL'] = load_system('PILL', MODELS['PILL'])
except Exception as e:
    print(f"CRITICAL INIT ERROR: {e}")

# ==========================================
# 📹 CORE PROCESSING
# ==========================================
def process_frame(frame):
    global ZOOM_LEVEL, CURRENT_MODE
    
    # Safety Check: ถ้า Frame ว่าง ให้คืนค่ากลับไปเลย
    if frame is None or frame.size == 0: 
        return np.zeros((480, 640, 3), dtype=np.uint8)

    system = loaded_systems.get(CURRENT_MODE)
    # ถ้าโหมดยังโหลดไม่เสร็จ ให้แสดงภาพสดไปก่อน
    if not system: 
        cv2.putText(frame, "System Loading / Model Not Found", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

    # ZOOM Logic
    if ZOOM_LEVEL > 1.0:
        h, w, _ = frame.shape
        new_w, new_h = int(w/ZOOM_LEVEL), int(h/ZOOM_LEVEL)
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

    display_frame = frame.copy()
    h_main, w_main, _ = display_frame.shape
    pill_crop_raw = None
    crop_coords = None

    # 1. SEGMENTATION INFERENCE
    if system.get('det_model'):
        results = system['det_model'](frame, verbose=False, imgsz=640, conf=0.5)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            max_idx = torch.argmax(boxes.conf).item()
            x1_raw, y1_raw, x2_raw, y2_raw = boxes.xyxy[max_idx].cpu().numpy()
            
            w_box, h_box = x2_raw - x1_raw, y2_raw - y1_raw
            pad_w = int(w_box * PADDING_RATIO)
            pad_h = int(h_box * PADDING_RATIO)
            
            x1 = int(max(0, x1_raw - pad_w))
            y1 = int(max(0, y1_raw - pad_h))
            x2 = int(min(w_main, x2_raw + pad_w))
            y2 = int(min(h_main, y2_raw + pad_h))
            
            pill_crop_raw = frame[y1:y2, x1:x2]
            crop_coords = (x1, y1, x2-x1, y2-y1)

    # 2. CLASSIFY
    class_name = "Scanning..."
    conf_val = 0.0
    color = (100, 100, 100)

    if pill_crop_raw is not None and pill_crop_raw.size > 0 and system.get('cls_model'):
        final_input_img = remove_green_bg_auto(pill_crop_raw)
        img_pil = Image.fromarray(cv2.cvtColor(final_input_img, cv2.COLOR_BGR2RGB))
        input_tensor = system['transform'](img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            emb = system['cls_model'](input_tensor)
            logits = F.linear(F.normalize(emb), F.normalize(system['cls_model'].head.weight))
            probs = F.softmax(logits * 30.0, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        conf_val = confidence.item()
        idx_val = predicted_idx.item()
        
        if conf_val > CONFIDENCE_THRESHOLD:
            class_name = system['class_names'][idx_val]
            color = (0, 255, 0)
        else:
            class_name = "Unknown"
            color = (0, 0, 255)

        if crop_coords:
            cx, cy, cw, ch = crop_coords
            cv2.rectangle(display_frame, (cx, cy), (cx+cw, cy+ch), color, 3)

    # 3. UI Overlay
    # Preview Box
    if 'final_input_img' in locals() and final_input_img is not None:
        try:
            preview = cv2.resize(final_input_img, (150, 150))
            preview = cv2.copyMakeBorder(preview, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
            h, w, _ = preview.shape
            display_frame[100:100+h, w_main-w-20:w_main-20] = preview
        except: pass

    # Status Bar
    cv2.rectangle(display_frame, (0, 0), (w_main, 40), (0,0,0), -1)
    cv2.putText(display_frame, f"MODE: {CURRENT_MODE} (Press 'P')", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Result Box
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 50), (350, 150), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
    
    cv2.putText(display_frame, f"CLASS: {class_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"CONF:  {conf_val*100:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return display_frame

def generate_frames():
    # ส่ง CAMERA_INDEX ที่หาเจอแล้วเข้าไป
    vs = WebcamStream(src=CAMERA_INDEX).start()
    
    # รอให้กล้อง warm up นิดนึง
    time.sleep(1.0)
    
    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.1)
            continue
            
        with lock:
            processed = process_frame(frame)
            
        try:
            ret, buffer = cv2.imencode('.jpg', processed)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Encode error: {e}")
            continue

    vs.stop()

# ==========================================
# 🌐 SERVER
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
    with lock: CURRENT_MODE = 'PILL' if CURRENT_MODE == 'BOX' else 'BOX'
    return jsonify(success=True, mode=CURRENT_MODE)

@app.route('/get_mode')
def get_mode(): return jsonify(mode=CURRENT_MODE)

if __name__ == '__main__':
    print("🌍 Server running: http://0.0.0.0:5000")
    # ปิด debug=True เพื่อป้องกันการ reloader loop ที่อาจแย่งกล้อง
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)