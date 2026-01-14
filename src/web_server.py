from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
import math
from collections import Counter

app = Flask(__name__)

# ==========================================
# ⚙️ GLOBAL CONFIG & STATE
# ==========================================
camera = None
ai_engine = None
config = None
current_mode = 'BOX'
zoom_level = 1.0
lock = threading.Lock()

# 🚀 HIGH FIDELITY SETTINGS (ชัดเป๊ะ)
SKIP_FRAMES = 3          # รัน AI ทุกๆ 3 เฟรม (ช่วย CPU นิดนึง)
STREAM_WIDTH = 1280      # ⚡️ ปรับกลับมาเป็น HD (จอใหญ่สะใจ)
JPEG_QUALITY = 90        # ⚡️ ภาพชัด 90% (ไม่แตกแล้ว)

# State Variables
qc_state = {'locked': False, 'winner_name': "...", 'vote_stats': "", 'avg_conf': 0.0}
frame_count = 0
qc_last_results = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PillTrack AI - HD</title>
    <style>
        body { background-color: #121212; color: white; font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; padding: 10px; }
        /* ปรับ CSS ให้รูปขยายเต็มจอที่สุดเท่าที่จะทำได้ */
        .container { 
            position: relative; 
            display: inline-block; 
            border: 2px solid #333; 
            border-radius: 8px; 
            overflow: hidden;
            width: 95%; /* กว้างเกือบเต็มจอ */
            max-width: 1280px; /* ไม่เกิน HD */
        }
        img { display: block; width: 100%; height: auto; }
        .btn { padding: 15px 30px; font-size: 18px; margin: 5px; cursor: pointer; background: #222; color: #aaa; border: 1px solid #444; border-radius: 6px; }
        .btn.active { background: #00d4ff; color: #000; font-weight: bold; border-color: #00d4ff; }
        .status-bar { margin-top: 10px; color: #ccc; font-size: 1.2em; }
    </style>
</head>
<body>
    <h2>💊 PillTrack AI <span style="font-size:0.6em; color:#00ff00">v2.0 HD</span></h2>
    <div class="container"><img src="/video_feed"></div>
    <div class="status-bar">MODE: <span id="mode-display" style="color:#ffeb3b">...</span></div>
    <div style="margin-top:15px">
        <button class="btn" id="btn-box" onclick="setMode('BOX')">📦 Box</button>
        <button class="btn" id="btn-pill" onclick="setMode('PILL')">💊 Pill</button>
        <button class="btn" id="btn-qc" onclick="setMode('QC')">🕵️ QC</button>
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
        document.addEventListener('keydown', e => {
            if(e.key.toLowerCase() === 'p') setMode('QC');
        });
        setInterval(syncState, 1000);
    </script>
</body>
</html>
"""

def draw_overlay(frame, mode, results):
    display = frame # วาดลงภาพ Original เลย
    h, w = display.shape[:2]
    
    # Header
    cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
    color_map = {'BOX': (0, 255, 255), 'PILL': (0, 255, 0), 'QC': (255, 0, 255)}
    cv2.putText(display, f"MODE: {mode}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map.get(mode, (255,255,255)), 3)

    if mode == 'BOX':
        for item in results.get('box_data', []):
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 3)
            cv2.putText(display, f"{item['name']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    elif mode == 'PILL':
        if results.get('coords'):
            x, y, w_rect, h_rect = results['coords']
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), (0, 255, 0), 4)
            cv2.putText(display, f"ID: {results['name']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    elif mode == 'QC':
        qc_res = results.get('qc_data', {})
        if qc_res:
            # 1. Draw Pack
            if qc_res.get('pack_coords'):
                bx1, by1, bx2, by2 = qc_res['pack_coords']
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 4)
                cv2.putText(display, f"PACK: {qc_res.get('pack_name', 'Scanning...')}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 2. Draw Pills
            for p_coord in qc_res.get('pills_coords', []):
                px1, py1, px2, py2 = p_coord
                p_color = (0, 255, 0) if qc_state['locked'] else (200, 200, 200)
                cv2.rectangle(display, (px1, py1), (px2, py2), p_color, 2)

            # 3. Info Panel (ตัวหนังสือใหญ่ขึ้น)
            cv2.rectangle(display, (0, 60), (800, 180), (0,0,0), -1)
            pill_name = qc_res.get('pill_name', 'Scanning...')
            count = qc_res.get('pill_count', 0)
            stats = qc_res.get('vote_stats', '')
            
            cv2.putText(display, f"Result: {pill_name} (Qty: {count})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if stats:
                cv2.putText(display, f"Votes: {stats}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return display

def generate_frames():
    global frame_count, qc_state, qc_last_results
    qc_last_results = {}

    while True:
        frame = camera.read()
        if frame is None: 
            time.sleep(0.01)
            continue
            
        # ถ้ากล้องมาใหญ่มากๆ (เกิน HD) ให้ย่อเหลือ HD เพื่อไม่ให้ Server แตก
        # แต่ถ้ากล้องเป็น HD อยู่แล้ว ก็ใช้เต็มๆ เลย
        h, w = frame.shape[:2]
        if w > 1920: 
            frame = cv2.resize(frame, (1280, 720))

        # Digital Zoom
        if zoom_level > 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: mode = current_mode

        if mode != 'QC': qc_state['locked'] = False

        # SKIP FRAMES Logic
        should_run_ai = (frame_count % SKIP_FRAMES == 0) or (not qc_last_results)
        
        if should_run_ai:
            if mode == 'BOX':
                boxes = ai_engine.predict_box_locations(frame)
                box_res = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad=10; h_img, w_img = frame.shape[:2]
                    bx1, by1 = max(0, x1-pad), max(0, y1-pad)
                    bx2, by2 = min(w_img, x2+pad), min(h_img, y2+pad)
                    crop = frame[by1:by2, bx1:bx2]
                    name, conf, _ = ai_engine.identify_object(crop, mode='BOX', preprocess='green_screen')
                    box_res.append({'coords':(x1,y1,x2,y2), 'name':name, 'conf':conf})
                results['box_data'] = box_res

            elif mode == 'PILL':
                boxes = ai_engine.predict_box_locations(frame)
                if len(boxes) > 0:
                    h_img, w_img = frame.shape[:2]; cx, cy = w_img//2, h_img//2
                    best_box = min(boxes, key=lambda b: math.hypot((b.xyxy[0][0]+b.xyxy[0][2])/2 - cx, (b.xyxy[0][1]+b.xyxy[0][3])/2 - cy))
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    pad=20; px1, py1 = max(0, x1-pad), max(0, y1-pad); px2, py2 = min(w_img, x2+pad), min(h_img, y2+pad)
                    crop = frame[py1:py2, px1:px2]
                    name, conf, _ = ai_engine.identify_object(crop, mode='PILL', preprocess='green_screen')
                    results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf})

            elif mode == 'QC':
                pack_boxes = ai_engine.predict_box_locations(frame)
                
                if not pack_boxes:
                    qc_state = {'locked': False, 'winner_name': "...", 'vote_stats': "", 'avg_conf': 0.0}
                    results['qc_data'] = {}
                else:
                    pack_boxes = sorted(pack_boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)
                    p_box = pack_boxes[0]
                    bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                    pack_crop = frame[by1:by2, bx1:bx2]
                    
                    # 1. Pack Name
                    pack_name, _, _ = ai_engine.identify_object(pack_crop, mode='BOX', preprocess='green_screen')
                    
                    # 2. Pills
                    pill_boxes = ai_engine.predict_pill_locations(pack_crop)
                    pills_coords = []
                    for pb in pill_boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        pills_coords.append((bx1+px1, by1+py1, bx1+px2, by1+py2))
                    
                    # 3. Vote
                    if not qc_state['locked'] and len(pill_boxes) > 0:
                        votes = []
                        for pb in pill_boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            pill_crop = pack_crop[py1:py2, px1:px2]
                            p_name, _, _ = ai_engine.identify_object(pill_crop, mode='PILL', preprocess='green_screen')
                            votes.append(p_name)
                        
                        if votes:
                            vote_counts = Counter(votes)
                            winner = vote_counts.most_common(1)[0][0]
                            stats_str = ", ".join([f"{k}:{v}" for k,v in vote_counts.items()])
                            qc_state['locked'] = True
                            qc_state['winner_name'] = winner
                            qc_state['vote_stats'] = stats_str
                    
                    results['qc_data'] = {
                        'pack_coords': (bx1, by1, bx2, by2),
                        'pack_name': pack_name,
                        'pills_coords': pills_coords,
                        'pill_count': len(pills_coords),
                        'pill_name': qc_state['winner_name'],
                        'vote_stats': qc_state['vote_stats']
                    }

            qc_last_results = results
        else:
            results = qc_last_results

        # Draw Overlay
        final_frame = draw_overlay(frame, mode, results)
        frame_count += 1
        
        # 🚀 FINAL OUTPUT: High Resolution & High Quality 🚀
        # ถ้าภาพต้นฉบับใหญ่ (เช่น 1280px) ก็ส่งไปเต็มๆ เลย หรือย่อเล็กน้อยเพื่อให้เน็ตวิ่งไหว
        # ปรับ STREAM_WIDTH ด้านบนได้ถ้าอยากได้ใหญ่กว่านี้
        h, w = final_frame.shape[:2]
        if w > STREAM_WIDTH:
            aspect_ratio = h / w
            target_h = int(STREAM_WIDTH * aspect_ratio)
            stream_frame = cv2.resize(final_frame, (STREAM_WIDTH, target_h))
        else:
            stream_frame = final_frame
        
        # JPEG Quality 90 (ชัดกริ๊บ)
        ret, buffer = cv2.imencode('.jpg', stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ... (Routes เหมือนเดิม) ...
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
    print(f"🌍 Web Interface running (HD Mode) at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)