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
current_mode = 'BOX' # Default
zoom_level = 1.0
lock = threading.Lock()

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
        .btn { padding: 15px 25px; font-size: 16px; margin: 5px; cursor: pointer; background: #333; color: white; border: 1px solid #555; border-radius: 5px; }
        .btn.active { background: #00d4ff; color: black; font-weight: bold; }
        #mode-display { font-weight: bold; color: yellow; font-size: 20px; }
    </style>
</head>
<body>
    <h1>💊 PillTrack AI</h1>
    <div class="container"><img src="/video_feed"></div>
    <div style="margin-top: 20px;">Current Mode: <span id="mode-display">LOADING...</span></div>
    <div>
        <button class="btn" id="btn-box" onclick="setMode('BOX')">📦 Box Count</button>
        <button class="btn" id="btn-pill" onclick="setMode('PILL')">💊 Single Pill</button>
        <button class="btn" id="btn-qc" onclick="setMode('QC')">🕵️ QC Inspection</button>
    </div>
    <script>
        function updateUI(mode) {
            document.getElementById('mode-display').innerText = mode;
            ['btn-box', 'btn-pill', 'btn-qc'].forEach(id => document.getElementById(id).className = 'btn');
            if(mode === 'BOX') document.getElementById('btn-box').className = 'btn active';
            if(mode === 'PILL') document.getElementById('btn-pill').className = 'btn active';
            if(mode === 'QC') document.getElementById('btn-qc').className = 'btn active';
        }
        function setMode(mode) { fetch('/set_mode/' + mode).then(r=>r.json()).then(d=>updateUI(d.mode)); }
        function syncState() { fetch('/get_mode').then(r=>r.json()).then(d=>updateUI(d.mode)); }
        setInterval(syncState, 1000);
    </script>
</body>
</html>
"""

def draw_overlay(frame, mode, results):
    display = frame.copy()
    h, w, _ = display.shape
    
    # Status
    cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(display, f"MODE: {mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Preview Box (PIP)
    preview_img = results.get('preview_img')
    if preview_img is not None:
        try:
            mini = cv2.resize(preview_img, (150, 150))
            cv2.rectangle(mini, (0,0), (150,150), (255,255,255), 2)
            display[50:200, w-170:w-20] = mini
        except: pass

    if mode == 'BOX':
        for item in results.get('box_data', []):
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(display, f"{item['name']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    elif mode == 'PILL':
        # วาดกรอบยาเดียว
        coords = results.get('coords')
        if coords:
            x, y, w_rect, h_rect = coords
            color = (0, 255, 0) if results['conf'] > 0.65 else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), color, 3)
            # Info Box
            cv2.rectangle(display, (10, 50), (300, 120), (0,0,0), -1)
            cv2.putText(display, f"{results['name']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"{results['conf']:.1%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    elif mode == 'QC':
        # วาดกล่องใหญ่ (แผงยา)
        for box in results.get('qc_boxes', []):
            bx1, by1, bx2, by2 = box['coords']
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(display, "Pack", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # วาดเม็ดยาข้างใน
            for pill in box['pills']:
                px1, py1, px2, py2 = pill['coords']
                # สีตามความมั่นใจ
                p_color = (0, 255, 0) if pill['conf'] > 0.6 else (0, 0, 255)
                cv2.rectangle(display, (px1, py1), (px2, py2), p_color, 1)
                # ชื่อยาทีละเม็ด (ตัวเล็กๆ)
                cv2.putText(display, pill['name'][:4], (px1, py1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, p_color, 1)

    return display

def generate_frames():
    while True:
        frame = camera.read()
        if frame is None: time.sleep(0.1); continue
        
        if zoom_level > 1.0:
            h, w, _ = frame.shape
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: mode = current_mode
        
        # --- LOGIC ---
        if mode == 'BOX':
            boxes = ai_engine.predict_box_locations(frame)
            box_res = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad=10
                bx1, by1 = max(0, x1-pad), max(0, y1-pad)
                bx2, by2 = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                crop = frame[by1:by2, bx1:bx2]
                name, conf, proc = ai_engine.identify_object(crop, mode='BOX', use_bg_removal=True)
                box_res.append({'coords':(x1,y1,x2,y2), 'name':name, 'conf':conf})
            results['box_data'] = box_res

        elif mode == 'PILL':
            # เหมือนเดิม (ขอละไว้เพื่อความกระชับ - ใช้โค้ดเดิมได้เลย)
            boxes = ai_engine.predict_box_locations(frame) # ใช้ Box หาเม็ดเดี่ยว
            if len(boxes) > 0:
                best = max(boxes, key=lambda b: b.conf.item())
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                name, conf, proc = ai_engine.identify_object(crop, mode='PILL', use_bg_removal=True)
                results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf, 'preview_img':proc})
            else:
                h, w, _ = frame.shape
                cx, cy = w//2, h//2
                crop = frame[cy-150:cy+150, cx-150:cx+150]
                name, conf, proc = ai_engine.identify_object(crop, mode='PILL', use_bg_removal=True)
                results.update({'coords':(cx-150,cy-150,300,300), 'name':name, 'conf':conf, 'preview_img':proc})

        elif mode == 'QC':
            # 🕵️ QC MODE: Box -> Pills -> Identify (No BG Remove)
            box_locs = ai_engine.predict_box_locations(frame)
            qc_data = []
            
            for box in box_locs:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Crop แผงยาออกมา
                pack_img = frame[y1:y2, x1:x2]
                
                # หาเม็ดยา "ในแผง" (Inception!)
                pill_locs = ai_engine.predict_pill_locations(pack_img)
                pills_found = []
                
                for p_box in pill_locs:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                    # Crop เม็ดยา (จากภาพแผง)
                    pill_crop = pack_img[py1:py2, px1:px2]
                    
                    # Identify โดย "ไม่ลบ BG" (use_bg_removal=False)
                    p_name, p_conf, _ = ai_engine.identify_object(pill_crop, mode='PILL', use_bg_removal=False)
                    
                    # แปลงพิกัดกลับไปเป็นพิกัดภาพใหญ่ (เพื่อวาด)
                    abs_x1, abs_y1 = x1 + px1, y1 + py1
                    abs_x2, abs_y2 = x1 + px2, y1 + py2
                    
                    pills_found.append({
                        'coords': (abs_x1, abs_y1, abs_x2, abs_y2),
                        'name': p_name,
                        'conf': p_conf
                    })
                
                qc_data.append({'coords': (x1, y1, x2, y2), 'pills': pills_found})
            
            results['qc_boxes'] = qc_data

        final_frame = draw_overlay(frame, mode, results)
        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes เหมือนเดิม...
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