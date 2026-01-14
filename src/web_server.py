from flask import Flask, Response, render_template_string, jsonify
import cv2
import time
import threading
import numpy as np
import math

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
        body { background-color: #121212; color: white; font-family: 'Segoe UI', sans-serif; text-align: center; margin: 0; padding: 20px; }
        .container { position: relative; display: inline-block; border: 3px solid #333; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        img { display: block; max-width: 100%; height: auto; }
        .btn { padding: 12px 24px; font-size: 16px; margin: 5px; cursor: pointer; background: #222; color: #aaa; border: 1px solid #444; border-radius: 6px; transition: 0.2s; }
        .btn:hover { background: #333; color: white; }
        .btn.active { background: #00d4ff; color: #000; font-weight: bold; border-color: #00d4ff; box-shadow: 0 0 10px rgba(0, 212, 255, 0.3); }
        #mode-display { font-weight: bold; color: #ffeb3b; font-size: 20px; text-transform: uppercase; }
        .status-bar { margin-top: 15px; font-size: 1.1em; color: #ccc; }
    </style>
</head>
<body>
    <h1>💊 PillTrack AI <span style="font-size:0.5em; color:#555">v1.2</span></h1>
    <div class="container"><img src="/video_feed"></div>
    <div class="status-bar">MODE: <span id="mode-display">LOADING...</span></div>
    <div style="margin-top: 15px;">
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
                // Cycle modes: BOX -> PILL -> QC -> BOX
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

def draw_overlay(frame, mode, results):
    display = frame.copy()
    h, w, _ = display.shape
    
    # 1. Top Bar Background
    cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
    
    # 2. Mode Indicator
    color_map = {'BOX': (0, 255, 255), 'PILL': (0, 255, 0), 'QC': (255, 0, 255)}
    theme_color = color_map.get(mode, (255, 255, 255))
    cv2.putText(display, f"MODE: {mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme_color, 2)

    # 3. Preview Image (PIP) - มุมขวาบน
    preview_img = results.get('preview_img')
    if preview_img is not None:
        try:
            # Resize ให้เป็นสี่เหลี่ยมจัตุรัสสวยๆ
            mini_size = 180
            mini = cv2.resize(preview_img, (mini_size, mini_size))
            
            # ใส่กรอบให้รู้ว่าเป็น Pill ที่เราโฟกัส
            border_color = (0, 255, 0) if results.get('pill_conf', 0) > 0.6 else (0, 0, 255)
            mini = cv2.copyMakeBorder(mini, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            
            # วางตำแหน่ง (ขวาบน)
            mh, mw, _ = mini.shape
            y_offset = 60 # ลงมาจากแถบดำนิดนึง
            x_offset = w - mw - 20
            display[y_offset:y_offset+mh, x_offset:x_offset+mw] = mini
            
            # Label บน PIP
            cv2.putText(display, "Scan Target", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        except Exception as e: 
            print(f"PIP Error: {e}")

    # --- DRAWING PER MODE ---
    
    if mode == 'BOX':
        for item in results.get('box_data', []):
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)
            
            # Label
            label = f"{item['name']} ({item['conf']:.0%})"
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
            
            # Result Text (Left Side)
            cv2.putText(display, f"PILL: {results['name']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(display, f"CONF: {results['conf']:.1%}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    elif mode == 'QC':
        qc_data = results.get('qc_data', [])
        
        # วาดกล่องผลลัพธ์รวมมุมซ้าย
        if qc_data:
            # เอาแค่ Pack ใหญ่สุดมาโชว์ Text
            main_pack = qc_data[0] 
            pack_name = main_pack['pack_name']
            pill_name = main_pack['pill_name']
            
            # 📝 Output Text Format: Pack = ..., Pills = ...
            cv2.putText(display, f"Pack = {pack_name}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display, f"Pill = {pill_name}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        for pack in qc_data:
            # 1. วาดกรอบ Pack
            bx1, by1, bx2, by2 = pack['coords']
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            cv2.putText(display, f"Pack: {pack['pack_name']}", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 2. วาดเม็ดยาทุกเม็ด (กรอบบางๆ สีเทา)
            for p_coord in pack['all_pills_coords']:
                px1, py1, px2, py2 = p_coord
                cv2.rectangle(display, (px1, py1), (px2, py2), (100, 100, 100), 1) # Gray box
            
            # 3. วาดเม็ดที่ถูกเลือก (Selected Pill) - กรอบหนาสีเขียว
            if pack.get('selected_pill_coords'):
                sx1, sy1, sx2, sy2 = pack['selected_pill_coords']
                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
                # ลากเส้นจากเม็ดไปหา Text (Optional: ทำให้ดูไฮเทค)
                cv2.line(display, (sx1, sy1), (bx1, by1), (0, 255, 0), 1)

    return display

def generate_frames():
    while True:
        frame = camera.read()
        if frame is None: time.sleep(0.1); continue
        
        # Digital Zoom
        if zoom_level > 1.0:
            h, w, _ = frame.shape
            new_w, new_h = int(w/zoom_level), int(h/zoom_level)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        results = {}
        with lock: mode = current_mode
        
        if mode == 'BOX':
            boxes = ai_engine.predict_box_locations(frame)
            box_res = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad=10
                bx1, by1 = max(0, x1-pad), max(0, y1-pad)
                bx2, by2 = min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                crop = frame[by1:by2, bx1:bx2]
                
                # Identify Box Name
                name, conf, proc = ai_engine.identify_object(crop, mode='BOX', use_bg_removal=True)
                box_res.append({'coords':(x1,y1,x2,y2), 'name':name, 'conf':conf})
            
            results['box_data'] = box_res

        elif mode == 'PILL':
            boxes = ai_engine.predict_box_locations(frame)
            if len(boxes) > 0:
                # Center Focus Logic
                h, w, _ = frame.shape
                cx, cy = w//2, h//2
                best_box = min(boxes, key=lambda b: math.hypot((b.xyxy[0][0]+b.xyxy[0][2])/2 - cx, (b.xyxy[0][1]+b.xyxy[0][3])/2 - cy))
                
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                pad=20
                px1, py1 = max(0, x1-pad), max(0, y1-pad)
                px2, py2 = min(w, x2+pad), min(h, y2+pad)
                crop = frame[py1:py2, px1:px2]
                
                name, conf, proc = ai_engine.identify_object(crop, mode='PILL', use_bg_removal=True)
                results.update({'coords':(x1,y1,x2-x1,y2-y1), 'name':name, 'conf':conf, 'preview_img':proc})
            else:
                # Fallback: Scan Center
                h, w, _ = frame.shape
                c_crop = frame[h//2-150:h//2+150, w//2-150:w//2+150]
                name, conf, proc = ai_engine.identify_object(c_crop, mode='PILL', use_bg_removal=True)
                results.update({'coords':(w//2-150,h//2-150,300,300), 'name':name, 'conf':conf, 'preview_img':proc})

        elif mode == 'QC':
            # 🕵️ QC MODE: Logic ใหม่!
            # 1. หา Pack (Container)
            pack_boxes = ai_engine.predict_box_locations(frame)
            qc_results = []
            
            # เอา Pack ที่ใหญ่ที่สุด (สมมติว่าหมอถืออันเดียว) หรือวนลูปก็ได้
            # เรียงตามขนาด (Area) มากไปน้อย
            pack_boxes = sorted(pack_boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]), reverse=True)

            for p_box in pack_boxes:
                bx1, by1, bx2, by2 = map(int, p_box.xyxy[0])
                pack_crop = frame[by1:by2, bx1:bx2]
                
                # 1. Identify PACK Name (เช่น Paracap)
                pack_name, pack_conf, _ = ai_engine.identify_object(pack_crop, mode='BOX', use_bg_removal=True)
                
                # 2. หา Pills "ข้างใน Pack"
                pill_locs = ai_engine.predict_pill_locations(pack_crop)
                
                all_pills_coords = []
                selected_pill_data = None
                
                # หาเม็ดที่อยู่ "ตรงกลาง Pack" มากที่สุด (The Representative)
                pack_h, pack_w, _ = pack_crop.shape
                pack_cx, pack_cy = pack_w // 2, pack_h // 2
                min_dist = float('inf')
                best_pill_box = None
                
                for pl in pill_locs:
                    px1, py1, px2, py2 = map(int, pl.xyxy[0])
                    # แปลงพิกัดกลับเป็น Global เพื่อวาด
                    g_px1, g_py1, g_px2, g_py2 = bx1+px1, by1+py1, bx1+px2, by1+py2
                    all_pills_coords.append((g_px1, g_py1, g_px2, g_py2))
                    
                    # คำนวณระยะห่างจาก Center ของ Pack
                    p_cx, p_cy = (px1+px2)//2, (py1+py2)//2
                    dist = math.hypot(p_cx - pack_cx, p_cy - pack_cy)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_pill_box = (px1, py1, px2, py2) # พิกัด Local
                        selected_pill_data = {'global_coords': (g_px1, g_py1, g_px2, g_py2)}

                # 3. Identify SELECTED Pill (แค่เม็ดเดียว!)
                pill_name = "Scanning..."
                pill_conf = 0.0
                pill_preview = None
                
                if best_pill_box:
                    px1, py1, px2, py2 = best_pill_box
                    # Crop เม็ดยาจาก Pack (ไม่ต้องตัด BG เพราะเป็นฟอยล์)
                    pill_crop = pack_crop[py1:py2, px1:px2]
                    
                    # Identify (QC Mode = No BG Removal)
                    pill_name, pill_conf, pill_preview = ai_engine.identify_object(pill_crop, mode='PILL', use_bg_removal=False)
                    
                    # เก็บรูป Preview เพื่อส่งไปวาดข้างนอก
                    results['preview_img'] = pill_preview
                    results['pill_conf'] = pill_conf

                qc_results.append({
                    'coords': (bx1, by1, bx2, by2),
                    'pack_name': pack_name,
                    'pill_name': pill_name,
                    'all_pills_coords': all_pills_coords,
                    'selected_pill_coords': selected_pill_data['global_coords'] if selected_pill_data else None
                })
                
                # Process แค่ Pack เดียวพอ (ตัวที่ใหญ่ที่สุด) เพื่อ Performance
                break 
                
            results['qc_data'] = qc_results

        final_frame = draw_overlay(frame, mode, results)
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
    camera = _camera
    ai_engine = _engine
    config = _config
    print("🌍 Web Interface running at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)