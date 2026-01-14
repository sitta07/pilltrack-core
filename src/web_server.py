from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np

app = Flask(__name__)

# Global Vars
camera = None
ai_engine = None
config = None
current_mode = 'BOX'
zoom_level = 1.0 # 🔥 เพิ่ม Zoom Level กลับมา
lock = threading.Lock()

# HTML เหมือนเดิม แต่เพิ่ม Key Listener 'P'
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PillTrack AI</title>
    <style>
        body { background-color: #121212; color: white; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
        .container { position: relative; display: inline-block; border: 3px solid #333; }
        img { display: block; max-width: 100%; height: auto; }
        .btn { padding: 15px 30px; font-size: 18px; margin: 10px; cursor: pointer; background: #333; color: white; border: 1px solid #555; }
        .btn.active { background: #00d4ff; color: black; }
        #mode-display { font-weight: bold; color: yellow; font-size: 20px; }
    </style>
</head>
<body>
    <h1>💊 PillTrack AI</h1>
    <div class="container"><img src="/video_feed"></div>
    <div style="margin-top: 20px;">Current Mode: <span id="mode-display">LOADING...</span></div>
    <div>
        <button class="btn" id="btn-box" onclick="setMode('BOX')">📦 Box</button>
        <button class="btn" id="btn-pill" onclick="setMode('PILL')">💊 Pill</button>
    </div>
    <script>
        function updateUI(mode) {
            document.getElementById('mode-display').innerText = mode;
            document.getElementById('btn-box').className = mode === 'BOX' ? 'btn active' : 'btn';
            document.getElementById('btn-pill').className = mode === 'PILL' ? 'btn active' : 'btn';
        }
        function setMode(mode) { fetch('/set_mode/' + mode).then(r=>r.json()).then(d=>updateUI(d.mode)); }
        function syncState() { fetch('/get_mode').then(r=>r.json()).then(d=>updateUI(d.mode)); }
        
        // 🔥 เพิ่มกด P เพื่อเปลี่ยนโหมดให้เหมือนเดิม
        document.addEventListener('keydown', e => {
            if(e.key.toLowerCase() === 'p') {
                const newMode = document.getElementById('mode-display').innerText === 'BOX' ? 'PILL' : 'BOX';
                setMode(newMode);
            }
        });
        setInterval(syncState, 1000);
    </script>
</body>
</html>
"""

def draw_overlay(frame, mode, results):
    display = frame.copy()
    h, w, _ = display.shape
    
    # 1. Status Bar
    cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
    color = (0, 255, 0) if mode == 'PILL' else (0, 255, 255)
    cv2.putText(display, f"MODE: {mode} (Press 'P')", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 2. Preview Box (Picture-in-Picture) 🔥 ฟีเจอร์ที่หายไป
    preview_img = results.get('preview_img')
    if preview_img is not None:
        try:
            # Resize ให้พอดีกรอบเล็ก
            mini = cv2.resize(preview_img, (150, 150))
            # ใส่กรอบสีตามโหมด
            mini = cv2.copyMakeBorder(mini, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
            mh, mw, _ = mini.shape
            # วางมุมขวาบน (ใต้ Status bar นิดนึง)
            display[50:50+mh, w-mw-20:w-20] = mini
        except: pass

    if mode == 'BOX':
        box_data = results.get('box_data', [])
        cv2.putText(display, f"Count: {len(box_data)}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for item in box_data:
            x1, y1, x2, y2 = item['coords']
            name = item['name']
            conf = item['conf']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(display, f"{name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    elif mode == 'PILL':
        name = results.get('name', 'Scanning...')
        conf = results.get('conf', 0.0)
        
        # Result Box
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 50), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        text_color = (0, 255, 0) if conf > 0.65 else (0, 0, 255)
        cv2.putText(display, f"{name}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        cv2.putText(display, f"Conf: {conf:.1%}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        coords = results.get('coords')
        if coords:
            x, y, w_rect, h_rect = coords
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), text_color, 3)

    return display

def generate_frames():
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.1); continue
        
        # 🔥 เพิ่ม ZOOM Logic กลับมา
        if zoom_level > 1.0:
            h, w, _ = frame.shape
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: mode = current_mode
        
        if mode == 'BOX':
            boxes = ai_engine.predict_box_locations(frame)
            box_results = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad = 10
                h, w, _ = frame.shape
                bx1, by1 = max(0, x1-pad), max(0, y1-pad)
                bx2, by2 = min(w, x2+pad), min(h, y2+pad)
                box_crop = frame[by1:by2, bx1:bx2]
                
                # identify_object now returns 3 values
                name, conf, processed = ai_engine.identify_object(box_crop, mode='BOX')
                box_results.append({'coords':(x1,y1,x2,y2), 'name':name, 'conf':conf})
            
            results['box_data'] = box_results

        elif mode == 'PILL':
            boxes = ai_engine.predict_box_locations(frame)
            if len(boxes) > 0:
                best_box = max(boxes, key=lambda b: b.conf.item())
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                pad = 20
                h, w, _ = frame.shape
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                crop = frame[y1:y2, x1:x2]
                
                name, conf, processed = ai_engine.identify_object(crop, mode='PILL')
                results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf, 'preview_img': processed})
            else:
                h, w, _ = frame.shape
                cx, cy, cw, ch = w//2, h//2, 400, 400
                crop = frame[cy-200:cy+200, cx-200:cx+200]
                name, conf, processed = ai_engine.identify_object(crop, mode='PILL')
                results.update({'coords':(cx-200,cy-200,400,400), 'name':name, 'conf':conf, 'preview_img': processed})

        final_frame = draw_overlay(frame, mode, results)
        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ... (Routes ข้างล่างเหมือนเดิม) ...
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
    camera = _camera
    ai_engine = _engine
    config = _config
    print("🌍 Web Interface running at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)