from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
import os
import torch

app = Flask(__name__)

# ==========================================
# ⚙️ GLOBAL PRODUCTION STATE
# ==========================================
camera = None
ai_engine = None
latest_frame = None
last_results = {}
current_mode = 'BOX'
lock = threading.Lock()
running = True

# Metrics
fps_cam = 0
fps_ai = 0

# ==========================================
# ✂️ PRE-PROCESSING: CROP & BG REMOVAL
# ==========================================
def remove_background(image):
    """ 
    ฟังก์ชันลบพื้นหลังสไตล์ Production:
    ใช้ GrabCut หรือ Simple Masking เพื่อกำจัด Noise รอบยา/กล่อง
    """
    if image is None or image.size == 0: return image
    
    # วิธีที่ 1: ใช้ Simple Thresholding (ถ้าพื้นหลังนิ่ง) 
    # หรือ วิธีที่ 2: ใช้ AI Engine Preprocess (Green Screen / Masking)
    # ในที่นี้เราจะส่งเข้า engine.identify_object ที่มีการทำ preprocess อยู่แล้ว
    return image

# ==========================================
# 🧠 AI WORKER THREAD (GPU Processing)
# ==========================================
def ai_worker():
    global last_results, fps_ai
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    torch.backends.cudnn.enabled = False 
    
    while ai_engine is None and running:
        time.sleep(0.5)
        
    prev_time = time.time()
    while running:
        frame_to_proc = None
        with lock:
            if latest_frame is not None:
                frame_to_proc = latest_frame.copy()
                mode = current_mode
        
        if frame_to_proc is None:
            time.sleep(0.01)
            continue
            
        try:
            results = {'boxes': [], 'debug_preview': None}
            if mode == 'BOX':
                boxes = ai_engine.predict_box_locations(frame_to_proc)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # ✂️ 1. CROP: ตัดมาเฉพาะส่วนที่ YOLO เจอ
                        padding = 15
                        crop = frame_to_proc[max(0, y1-padding):y2+padding, max(0, x1-padding):x2+padding]
                        if crop.size == 0: continue
                        
                        # 🧼 2. IDENTIFY + BG REMOVAL: 
                        # ส่งเข้า Engine พร้อมสั่งลบพื้นหลัง (ถ้าคุณมีฟังก์ชัน green_screen ในนั้น)
                        name, conf, processed_crop = ai_engine.identify_object(crop, mode='BOX', preprocess='green_screen')
                        
                        results['boxes'].append({
                            'coords': (x1, y1, x2, y2), 
                            'name': name, 
                            'conf': conf
                        })
                        
                        # เก็บรูปที่ลบพื้นหลังแล้วมาโชว์ที่หน้าเว็บ (Debug)
                        if results['debug_preview'] is None:
                            results['debug_preview'] = processed_crop

            with lock:
                last_results = results
            
            curr_time = time.time()
            fps_ai = (fps_ai * 0.9) + (0.1 * (1.0 / (curr_time - prev_time)))
            prev_time = curr_time
        except Exception as e:
            print(f"⚠️ AI Worker Error: {e}")
            
        time.sleep(0.001)

# ==========================================
# 📹 VIDEO GENERATOR (Streaming)
# ==========================================
def generate_frames():
    global fps_cam, latest_frame
    prev_time = time.time()
    
    while running:
        if camera is None:
            time.sleep(0.1)
            continue
            
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        with lock:
            latest_frame = frame.copy()
            results = last_results.copy()
            mode = current_mode
        
        display = frame.copy()
        
        # 🎨 DRAW OVERLAY
        cv2.rectangle(display, (0, 0), (display.shape[1], 60), (20, 20, 20), -1)
        status_text = f"CAM: {int(fps_cam)} | AI: {int(fps_ai)} FPS [ACTIVE]"
        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        
        if 'boxes' in results:
            for b in results['boxes']:
                x1, y1, x2, y2 = b['coords']
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 120), 4)
                label = f"{b['name']} ({int(b['conf']*100)}%)"
                cv2.putText(display, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)

            # 🖼️ SHOW CROP & BG REMOVED PREVIEW (PIP)
            if results.get('debug_preview') is not None:
                try:
                    debug_img = results['debug_preview']
                    debug_img = cv2.resize(debug_img, (180, 180))
                    h_d, w_d = display.shape[:2]
                    display[70:250, w_d-200:w_d-20] = debug_img
                    cv2.rectangle(display, (w_d-200, 70), (w_d-20, 250), (0, 212, 255), 2)
                    cv2.putText(display, "AI VISION", (w_d-200, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 1)
                except: pass

        # Resizing for Web Performance
        web_display = cv2.resize(display, (854, 480))
        ret, buffer = cv2.imencode('.jpg', web_display, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret: continue
        
        curr_time = time.time()
        fps_cam = (fps_cam * 0.9) + (0.1 * (1.0 / (curr_time - prev_time)))
        prev_time = curr_time
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

# ==========================================
# 🩺 NURSE STATION UI
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>PillTrack AI Station</title>
    <style>
        body { background: #121212; color: #fff; font-family: sans-serif; text-align: center; margin: 0; padding: 10px; }
        .stream-card { border: 4px solid #333; border-radius: 20px; overflow: hidden; background: #000; display: inline-block; max-width: 950px; width: 100%; }
        .btn-group { display: flex; justify-content: center; gap: 20px; margin-top: 20px; padding: 0 10px; }
        .btn { flex: 1; padding: 25px; font-size: 24px; font-weight: bold; border-radius: 15px; border: none; cursor: pointer; transition: 0.3s; }
        .btn-box { background: #222; color: #888; }
        .btn-box.active { background: #00d1b2; color: #000; box-shadow: 0 0 20px rgba(0,209,178,0.4); }
        .btn-pill { background: #222; color: #888; }
        .btn-pill.active { background: #ffdd57; color: #000; box-shadow: 0 0 20px rgba(255,221,87,0.4); }
    </style>
</head>
<body>
    <h2 style="color: #00d1b2;">🩺 PILLTRACK STATION</h2>
    <div class="stream-card"><img src="/video_feed" style="width: 100%;"></div>
    <div class="btn-group">
        <button class="btn btn-box" id="btn-box" onclick="setMode('BOX')">📦 ตรวจสอบกล่อง</button>
        <button class="btn btn-pill" id="btn-pill" onclick="setMode('PILL')">💊 ตรวจสอบเม็ด</button>
    </div>
    <script>
        function setMode(m) {
            fetch('/set_mode/'+m).then(r=>r.json()).then(d=>{
                document.querySelectorAll('.btn').forEach(b=>b.classList.remove('active'));
                document.getElementById('btn-'+m.toLowerCase()).classList.add('active');
            });
        }
        setMode('BOX');
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode_route(mode):
    global current_mode
    with lock: current_mode = mode
    return jsonify(mode=current_mode)

@app.route('/get_mode')
def get_mode_route(): return jsonify(mode=current_mode)

def start_server(_camera, _engine, _config):
    global camera, ai_engine, config
    camera = _camera; ai_engine = _engine; config = _config
    threading.Thread(target=ai_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)