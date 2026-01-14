from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
from src.utils import get_auto_hsv_bounds, remove_green_bg_auto # เดี๋ยวเราจะเพิ่มฟังก์ชันนี้ใน utils ทีหลัง

app = Flask(__name__)

# Global Vars (Controller State)
camera = None
ai_engine = None
config = None
current_mode = 'BOX' # Default Mode
lock = threading.Lock()

# HTML Template (Minimal & Modern)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PillTrack AI</title>
    <style>
        body { background-color: #121212; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { margin-bottom: 10px; color: #00d4ff; }
        .container { position: relative; display: inline-block; border: 3px solid #333; border-radius: 10px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
        img { display: block; max-width: 100%; height: auto; }
        .controls { margin-top: 20px; display: flex; justify-content: center; gap: 20px; }
        .btn { padding: 15px 30px; font-size: 18px; border: none; border-radius: 8px; cursor: pointer; transition: 0.3s; font-weight: bold; }
        .btn-mode { background-color: #333; color: white; border: 2px solid #555; }
        .btn-mode.active { background-color: #00d4ff; color: #000; border-color: #00d4ff; box-shadow: 0 0 15px #00d4ff; }
        .status-bar { margin-top: 15px; font-size: 1.2rem; color: #aaa; }
        #mode-display { font-weight: bold; color: #ffeb3b; }
    </style>
</head>
<body>
    <h1>💊 PillTrack AI System</h1>
    <div class="container">
        <img src="/video_feed" alt="Video Stream">
    </div>
    
    <div class="status-bar">
        Current Mode: <span id="mode-display">LOADING...</span>
    </div>

    <div class="controls">
        <button class="btn btn-mode" id="btn-box" onclick="setMode('BOX')">📦 Box Counter</button>
        <button class="btn btn-mode" id="btn-pill" onclick="setMode('PILL')">💊 Pill ID</button>
    </div>

    <script>
        function updateUI(mode) {
            document.getElementById('mode-display').innerText = mode;
            document.getElementById('btn-box').className = mode === 'BOX' ? 'btn btn-mode active' : 'btn btn-mode';
            document.getElementById('btn-pill').className = mode === 'PILL' ? 'btn btn-mode active' : 'btn btn-mode';
        }

        function setMode(mode) {
            fetch('/set_mode/' + mode).then(r => r.json()).then(data => {
                updateUI(data.mode);
            });
        }

        function syncState() {
            fetch('/get_mode').then(r => r.json()).then(data => {
                updateUI(data.mode);
            });
        }
        
        // Sync every 1 second
        setInterval(syncState, 1000);
    </script>
</body>
</html>
"""

def draw_overlay(frame, mode, results):
    """ฟังก์ชันวาดผลลัพธ์ลงบนภาพ (แยก Logic การวาดออกมา)"""
    display = frame.copy()
    h, w, _ = display.shape
    
    # 1. วาด Mode Indicator มุมบนซ้าย
    cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
    color = (0, 255, 0) if mode == 'PILL' else (255, 165, 0)
    cv2.putText(display, f"MODE: {mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if mode == 'BOX':
        # วาดกล่องที่เจอ
        boxes = results.get('boxes', [])
        cv2.putText(display, f"Count: {len(boxes)}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 3)

    elif mode == 'PILL':
        # วาดผล Classify ยา
        name = results.get('name', 'Scanning...')
        conf = results.get('conf', 0.0)
        
        # กล่องผลลัพธ์โปร่งแสง
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 50), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        # ข้อความ
        text_color = (0, 255, 0) if conf > config['model']['pill_classifier']['conf_threshold'] else (100, 100, 255)
        cv2.putText(display, f"{name}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        cv2.putText(display, f"Conf: {conf:.1%}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # วาดกรอบรอบยา (ถ้ามี crop coords)
        coords = results.get('coords')
        if coords:
            x, y, w, h = coords
            cv2.rectangle(display, (x, y), (x+w, y+h), text_color, 2)

    return display

def generate_frames():
    """Generator Function สำหรับ Flask Video Stream"""
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # --- AI PROCESSING LOGIC ---
        results = {}
        processed_frame = frame
        
        with lock:
            mode = current_mode
        
        if mode == 'BOX':
            boxes = ai_engine.predict_box(frame)
            results['boxes'] = boxes
            
        elif mode == 'PILL':
            # Logic ตัดภาพยา (Simplified)
            # 1. ใช้ Box Detector หาเม็ดยาก่อน (ถ้ามี)
            boxes = ai_engine.predict_box(frame)
            if len(boxes) > 0:
                # เอาเม็ดที่ Conf สูงสุด
                best_box = max(boxes, key=lambda b: b.conf.item())
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Padding
                pad = 20
                h, w, _ = frame.shape
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                
                crop = frame[y1:y2, x1:x2]
                results['coords'] = (x1, y1, x2-x1, y2-y1)
                
                # Predict
                name, conf = ai_engine.predict_pill(crop)
                results['name'] = name
                results['conf'] = conf
            else:
                # ถ้าไม่เจอกล่อง ลอง Predict ทั้งภาพ (หรือ Center crop)
                h, w, _ = frame.shape
                cx, cy = w//2, h//2
                cw, ch = 400, 400
                crop = frame[cy-200:cy+200, cx-200:cx+200]
                results['coords'] = (cx-200, cy-200, 400, 400)
                name, conf = ai_engine.predict_pill(crop)
                results['name'] = name
                results['conf'] = conf

        # วาดทับ
        final_frame = draw_overlay(frame, mode, results)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode_route(mode):
    global current_mode
    if mode in ['BOX', 'PILL']:
        with lock:
            current_mode = mode
    return jsonify(mode=current_mode)

@app.route('/get_mode')
def get_mode_route():
    return jsonify(mode=current_mode)

def start_server(_camera, _engine, _config):
    global camera, ai_engine, config
    camera = _camera
    ai_engine = _engine
    config = _config
    
    # Run Flask (Block Thread)
    print("🌍 Web Interface running at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)