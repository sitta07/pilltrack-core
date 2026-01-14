from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
import math
from collections import Counter

app = Flask(__name__)

# ==========================================
# ⚙️ GLOBAL STATE & CONFIG
# ==========================================
camera = None
ai_engine = None
config = None
current_mode = 'BOX'
zoom_level = 1.0
lock = threading.Lock()

# 🚀 QC STATE MACHINE
# เก็บสถานะการโหวต
qc_state = {
    'locked': False,        # ล็อคผลรึยัง?
    'winner_name': "...",   # ชื่อยาที่ชนะโหวต
    'vote_stats': "",       # สถิติการโหวต (เช่น Para:7, Sara:1)
    'avg_conf': 0.0         # ความมั่นใจเฉลี่ย
}

# Optimization Vars
frame_count = 0
SKIP_FRAMES = 3  # รัน AI ทุก 3 เฟรม

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PillTrack AI</title>
    <style>
        body { background-color: #121212; color: white; font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; padding: 20px; }
        .container { position: relative; display: inline-block; border: 3px solid #333; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        img { display: block; max-width: 100%; height: auto; }
        .btn { padding: 12px 24px; font-size: 16px; margin: 5px; cursor: pointer; background: #222; color: #aaa; border: 1px solid #444; border-radius: 6px; transition: 0.2s; }
        .btn:hover { background: #333; color: white; }
        .btn.active { background: #00d4ff; color: #000; font-weight: bold; border-color: #00d4ff; box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
        #mode-display { font-weight: bold; color: #ffeb3b; font-size: 20px; text-transform: uppercase; }
        .status-bar { margin-top: 15px; font-size: 1.1em; color: #ccc; }
        .controls { margin-top: 15px; }
    </style>
</head>
<body>
    <h1>💊 PillTrack AI <span style="font-size:0.5em; color:#555">v1.8 QC-Vote</span></h1>
    <div class="container"><img src="/video_feed"></div>
    <div class="status-bar">MODE: <span id="mode-display">LOADING...</span></div>
    <div class="controls">
        <button class="btn" id="btn-box" onclick="setMode('BOX')">📦 Box Count</button>
        <button class="btn" id="btn-pill" onclick="setMode('PILL')">💊 Single Pill</button>
        <button class="btn" id="btn-qc" onclick="setMode('QC')">🕵️ QC Inspect</button>
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
            if(e.key.toLowerCase() === 'p') {
                const modes = ['BOX', 'PILL', 'QC'];
                const current = document.getElementById('mode-display').innerText;
                setMode(modes[(modes.indexOf(current) + 1) % modes.length]);
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
    
    # Top Bar
    cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
    color_map = {'BOX': (0, 255, 255), 'PILL': (0, 255, 0), 'QC': (255, 0, 255)}
    cv2.putText(display, f"MODE: {mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_map.get(mode, (255,255,255)), 2)

    # PIP Preview (เฉพาะ Single Pill Mode)
    if mode == 'PILL' and results.get('preview_img') is not None:
        try:
            mini = cv2.resize(results['preview_img'], (150, 150))
            display[70:220, w-170:w-20] = mini
            cv2.rectangle(display, (w-170, 70), (w-20, 220), (0, 255, 0), 2)
        except: pass

    # --- DRAWING ---
    if mode == 'BOX':
        for item in results.get('box_data', []):
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            label = f"{item['name']} {item['conf']:.0%}"
            cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    elif mode == 'PILL':
        if results.get('coords'):
            x, y, w_rect, h_rect = results['coords']
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), (0, 255, 0), 3)
            cv2.putText(display, f"ID: {results['name']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    elif mode == 'QC':
        qc_res = results.get('qc_data', {})
        if qc_res:
            # 1. วาด Pack (Pack ทำงานตลอดเวลา)
            if qc_res.get('pack_coords'):
                bx1, by1, bx2, by2 = qc_res['pack_coords']
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
                
                # Show Pack Name
                pack_text = f"PACK: {qc_res.get('pack_name', 'Scanning...')}"
                cv2.putText(display, pack_text, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 2. วาด Pills (Loop วาดทุกเม็ดที่เจอ)
            for p_coord in qc_res.get('pills_coords', []):
                px1, py1, px2, py2 = p_coord
                # สีเขียว = Locked แล้ว, สีขาว = กำลัง Scan
                p_color = (0, 255, 0) if qc_state['locked'] else (200, 200, 200)
                cv2.rectangle(display, (px1, py1), (px2, py2), p_color, 2)

            # 3. Text Info (Output 2 บรรทัดตามบรีฟ)
            # พื้นหลัง Text
            cv2.rectangle(display, (0, 60), (600, 160), (0,0,0), -1)
            
            # Line 1: Pack Name
            cv2.putText(display, f"Pack: {qc_res.get('pack_name', 'Unknown')}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Line 2: Pill Name + Stats
            # Format: Pill: [Name] (Found: X | Para: Y, Sara: Z)
            pill_name = qc_res.get('pill_name', 'Scanning...')
            pill_count = qc_res.get('pill_count', 0)
            stats = qc_res.get('vote_stats', '')
            
            info_str = f"Pill: {pill_name} (Found: {pill_count} | {stats})"
            cv2.putText(display, info_str, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return display

def generate_frames():
    global frame_count, qc_state
    qc_last_results = {} # Local Cache

    while True:
        frame = camera.read()
        if frame is None: time.sleep(0.01); continue
        
        if zoom_level > 1.0:
            h, w, _ = frame.shape
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: mode = current_mode

        # Reset QC State if mode changed
        if mode != 'QC': 
            qc_state['locked'] = False

        should_run_ai = (frame_count % SKIP_FRAMES == 0)
        
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
                # Single Pill Logic (Same as before)
                boxes = ai_engine.predict_box_locations(frame)
                if len(boxes) > 0:
                    h_img, w_img = frame.shape[:2]; cx, cy = w_img//2, h_img//2
                    best_box = min(boxes, key=lambda b: math.hypot((b.xyxy[0][0]+b.xyxy[0][2])/2 - cx, (b.xyxy[0][1]+b.xyxy[0][3])/2 - cy))
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    pad=20; px1, py1 = max(0, x1-pad), max(0, y1-pad); px2, py2 = min(w_img, x2+pad), min(h_img, y2+pad)
                    crop = frame[py1:py2, px1:px2]
                    name, conf, proc = ai_engine.identify_object(crop, mode='PILL', preprocess='green_screen')
                    results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf, 'preview_img':proc})

            elif mode == 'QC':
                # 🔥 HYBRID QC LOGIC 🔥
                
                # 1. PACK DETECTION (Run Always) -> เพื่อให้กรอบเหลืองขยับได้ตลอด
                pack_boxes = ai_engine.predict_box_locations(frame)
                
                if not pack_boxes:
                    # ถ้าไม่เจอแผง -> Reset Vote
                    qc_state = {'locked': False, 'winner_name': "...", 'vote_stats': "", 'avg_conf': 0.0}
                    results['qc_data'] = {}
                else:
                    # เลือกแผงใหญ่สุด
                    pack_boxes = sorted(pack_boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)
                    p_box = pack_boxes[0]
                    bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                    pack_crop = frame[by1:by2, bx1:bx2]
                    
                    # Identify Pack Name (Real-time)
                    pack_name, _, _ = ai_engine.identify_object(pack_crop, mode='BOX', preprocess='green_screen')
                    
                    # 2. PILL DETECTION (YOLO Only) -> เพื่อเอาจำนวนและตำแหน่ง (Run Always)
                    pill_boxes = ai_engine.predict_pill_locations(pack_crop)
                    pills_coords = []
                    for pb in pill_boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        pills_coords.append((bx1+px1, by1+py1, bx1+px2, by1+py2)) # Global Coords
                    
                    # 3. VOTING LOGIC (Run Once per Pack Session)
                    if not qc_state['locked'] and len(pill_boxes) > 0:
                        votes = []
                        for pb in pill_boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            pill_crop = pack_crop[py1:py2, px1:px2]
                            # Identify Pill (Back to Basic: Green Screen)
                            p_name, _, _ = ai_engine.identify_object(pill_crop, mode='PILL', preprocess='green_screen')
                            votes.append(p_name)
                        
                        if votes:
                            vote_counts = Counter(votes)
                            winner = vote_counts.most_common(1)[0][0]
                            stats_str = ", ".join([f"{k}:{v}" for k,v in vote_counts.items()])
                            
                            # LOCK RESULT! 🔒
                            qc_state['locked'] = True
                            qc_state['winner_name'] = winner
                            qc_state['vote_stats'] = stats_str
                    
                    # Pack Info & Results
                    results['qc_data'] = {
                        'pack_coords': (bx1, by1, bx2, by2),
                        'pack_name': pack_name,          # Update ตลอด
                        'pills_coords': pills_coords,    # Update ตลอด (กรอบขยับตาม)
                        'pill_count': len(pills_coords), # จำนวน Update ตลอด
                        'pill_name': qc_state['winner_name'], # Locked Value
                        'vote_stats': qc_state['vote_stats']  # Locked Value
                    }

            qc_last_results = results
        else:
            results = qc_last_results

        final_frame = draw_overlay(frame, mode, results)
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
    print("🌍 Web Interface running at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)