from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
import math

app = Flask(__name__)

# ==========================================
# ⚙️ GLOBAL STATE & CONFIG
# ==========================================
camera = None
ai_engine = None
config = None
current_mode = 'BOX' # Default Mode
zoom_level = 1.0
lock = threading.Lock()

# 🚀 OPTIMIZATION VARS
qc_last_results = {}     # จำผลลัพธ์ล่าสุด
frame_count = 0          # ตัวนับเฟรม
SKIP_FRAMES = 3          # รัน AI ใหม่ทุกๆ 3 เฟรม (ลดภาระ CPU 66%)

# ==========================================
# 🎨 UI TEMPLATE
# ==========================================
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
    <h1>💊 PillTrack AI <span style="font-size:0.5em; color:#555">v1.2 Pro</span></h1>
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
                let next = modes[(modes.indexOf(current) + 1) % modes.length];
                setMode(next);
            }
        });
        setInterval(syncState, 1000);
    </script>
</body>
</html>
"""

# ==========================================
# 🖌️ DRAWING ENGINE
# ==========================================
def draw_overlay(frame, mode, results):
    display = frame.copy()
    h, w, _ = display.shape
    
    # 1. Header Bar
    cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
    
    # 2. Mode Title
    color_map = {'BOX': (0, 255, 255), 'PILL': (0, 255, 0), 'QC': (255, 0, 255)}
    theme_color = color_map.get(mode, (255, 255, 255))
    cv2.putText(display, f"MODE: {mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme_color, 2)

    # 3. PIP Preview (Picture-in-Picture)
    preview_img = results.get('preview_img')
    if preview_img is not None:
        try:
            mini_size = 180
            mini = cv2.resize(preview_img, (mini_size, mini_size))
            
            # Border color based on confidence
            conf = results.get('pill_conf', results.get('conf', 0))
            border_color = (0, 255, 0) if conf > 0.6 else (0, 0, 255)
            mini = cv2.copyMakeBorder(mini, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            
            mh, mw, _ = mini.shape
            y_off, x_off = 60, w - mw - 20
            display[y_off:y_off+mh, x_off:x_off+mw] = mini
            
            cv2.putText(display, "Target", (x_off, y_off - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        except: pass

    # --- MODE SPECIFIC DRAWING ---
    if mode == 'BOX':
        for item in results.get('box_data', []):
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            
            label = f"{item['name']} {item['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display, (x1, y1-25), (x1+tw, y1), (255, 165, 0), -1)
            cv2.putText(display, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    elif mode == 'PILL':
        coords = results.get('coords')
        if coords:
            x, y, w_rect, h_rect = coords
            color = (0, 255, 0) if results['conf'] > 0.65 else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x+w_rect, y+h_rect), color, 3)
            # Crosshair
            cx, cy = x + w_rect//2, y + h_rect//2
            cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            
            # Info Text
            cv2.putText(display, f"PILL: {results['name']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display, f"CONF: {results['conf']:.1%}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    elif mode == 'QC':
        qc_data = results.get('qc_data', [])
        
        # Summary Text (Left Side)
        if qc_data:
            main_pack = qc_data[0]
            cv2.putText(display, f"PACK: {main_pack['pack_name']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, f"PILL: {main_pack['pill_name']}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for pack in qc_data:
            # Pack Box
            bx1, by1, bx2, by2 = pack['coords']
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            cv2.putText(display, "Pack", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # All Pills (Gray)
            for p_coord in pack['all_pills_coords']:
                px1, py1, px2, py2 = p_coord
                cv2.rectangle(display, (px1, py1), (px2, py2), (100, 100, 100), 1)
            
            # Selected Pill (Green)
            if pack.get('selected_pill_coords'):
                sx1, sy1, sx2, sy2 = pack['selected_pill_coords']
                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)

    return display

# ==========================================
# 📹 CORE LOOP (OPTIMIZED)
# ==========================================
def generate_frames():
    global frame_count, qc_last_results
    
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        # 1. Digital Zoom
        if zoom_level > 1.0:
            h, w, _ = frame.shape
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: 
            mode = current_mode
        
        # 2. Optimization Logic (Skip Frames)
        # รัน AI ใหม่ก็ต่อเมื่อ: ครบกำหนดรอบ OR เปลี่ยนโหมด OR ยังไม่มี Cache
        should_run_ai = (frame_count % SKIP_FRAMES == 0)
        if 'mode' not in qc_last_results or qc_last_results['mode'] != mode:
            should_run_ai = True 

        if should_run_ai:
            # 🔥 HEAVY AI PROCESSING 🔥
            if mode == 'BOX':
                boxes = ai_engine.predict_box_locations(frame)
                box_res = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad=10
                    h_img, w_img, _ = frame.shape
                    bx1, by1 = max(0, x1-pad), max(0, y1-pad)
                    bx2, by2 = min(w_img, x2+pad), min(h_img, y2+pad)
                    crop = frame[by1:by2, bx1:bx2]
                    
                    name, conf, _ = ai_engine.identify_object(crop, mode='BOX', use_bg_removal=True)
                    box_res.append({'coords':(x1,y1,x2,y2), 'name':name, 'conf':conf})
                results['box_data'] = box_res

            elif mode == 'PILL':
                boxes = ai_engine.predict_box_locations(frame)
                if len(boxes) > 0:
                    h_img, w_img, _ = frame.shape
                    cx, cy = w_img//2, h_img//2
                    # Find box closest to center
                    best_box = min(boxes, key=lambda b: math.hypot((b.xyxy[0][0]+b.xyxy[0][2])/2 - cx, (b.xyxy[0][1]+b.xyxy[0][3])/2 - cy))
                    
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    pad=20
                    px1, py1 = max(0, x1-pad), max(0, y1-pad)
                    px2, py2 = min(w_img, x2+pad), min(h_img, y2+pad)
                    crop = frame[py1:py2, px1:px2]
                    
                    name, conf, proc = ai_engine.identify_object(crop, mode='PILL', use_bg_removal=True)
                    results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf, 'preview_img':proc})
                else:
                    # Fallback Center
                    h_img, w_img, _ = frame.shape
                    c_crop = frame[h_img//2-150:h_img//2+150, w_img//2-150:w_img//2+150]
                    name, conf, proc = ai_engine.identify_object(c_crop, mode='PILL', use_bg_removal=True)
                    results.update({'coords':(w_img//2-150,h_img//2-150,300,300), 'name':name, 'conf':conf, 'preview_img':proc})

            elif mode == 'QC':
                pack_boxes = ai_engine.predict_box_locations(frame)
                qc_data = []
                # Sort largest pack first
                pack_boxes = sorted(pack_boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)

                for p_box in pack_boxes:
                    bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                    pack_crop = frame[by1:by2, bx1:bx2]
                    
                    # A. Identify Pack
                    pack_name, pack_conf, _ = ai_engine.identify_object(pack_crop, mode='BOX', use_bg_removal=True)
                    
                    # B. Find Pills Inside
                    pill_locs = ai_engine.predict_pill_locations(pack_crop)
                    all_pills_coords = []
                    selected_pill_data = None
                    
                    pack_h, pack_w, _ = pack_crop.shape
                    pack_cx, pack_cy = pack_w // 2, pack_h // 2
                    min_dist = float('inf')
                    best_pill_box = None
                    
                    for pl in pill_locs:
                        px1, py1, px2, py2 = map(int, pl.xyxy[0])
                        g_px1, g_py1, g_px2, g_py2 = bx1+px1, by1+py1, bx1+px2, by1+py2
                        all_pills_coords.append((g_px1, g_py1, g_px2, g_py2))
                        
                        # Find pill closest to pack center
                        p_cx, p_cy = (px1+px2)//2, (py1+py2)//2
                        dist = math.hypot(p_cx - pack_cx, p_cy - pack_cy)
                        if dist < min_dist:
                            min_dist = dist
                            best_pill_box = (px1, py1, px2, py2)
                            selected_pill_data = {'global_coords': (g_px1, g_py1, g_px2, g_py2)}

                    # C. Identify Selected Pill
                    pill_name = "Scanning..."
                    pill_conf = 0.0
                    pill_preview = None
                    
                    if best_pill_box:
                        px1, py1, px2, py2 = best_pill_box
                        pill_crop = pack_crop[py1:py2, px1:px2]
                        # QC Mode = No BG Removal
                        pill_name, pill_conf, pill_preview = ai_engine.identify_object(pill_crop, mode='PILL', use_bg_removal=False)
                        results['preview_img'] = pill_preview
                        results['pill_conf'] = pill_conf

                    qc_data.append({
                        'coords': (bx1, by1, bx2, by2),
                        'pack_name': pack_name,
                        'pill_name': pill_name,
                        'all_pills_coords': all_pills_coords,
                        'selected_pill_coords': selected_pill_data['global_coords'] if selected_pill_data else None
                    })
                    break # Process only the largest pack
                
                results['qc_data'] = qc_data
            
            # Update Cache
            results['mode'] = mode
            qc_last_results = results
        
        else:
            # 💤 CACHE REUSE (Lightweight)
            results = qc_last_results
            if 'preview_img' in qc_last_results:
                results['preview_img'] = qc_last_results['preview_img']

        # 3. Draw Overlay
        final_frame = draw_overlay(frame, mode, results)
        frame_count += 1
        
        # 4. Yield Frame
        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# 🌐 ROUTES & SERVER START
# ==========================================
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