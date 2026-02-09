import cv2
import time
import threading
import torch
import os
import numpy as np
from ultralytics import YOLO

# ==========================================
# ⚙️ CONFIGURATIONS
# ==========================================
MODEL_PATH = "models/box_detector.onnx" # เช็ค Path ให้ถูกนะ
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720

# Global State
latest_frame = None
results_ai = []
fps_camera = 0
fps_ai = 0
running = True
lock = threading.Lock()

# RTX 50 Blackwell Stability Fix
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# ==========================================
# 📹 CAMERA THREAD (ดึงภาพอย่างเดียว)
# ==========================================
def camera_worker():
    global latest_frame, fps_camera, running
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60) # พยายามดันที่ 60 FPS
    
    prev_time = time.time()
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # คำนวณ FPS กล้อง
        curr_time = time.time()
        diff = curr_time - prev_time
        if diff > 0:
            fps_camera = (fps_camera * 0.9) + (0.1 * (1.0 / diff))
        prev_time = curr_time

        with lock:
            latest_frame = frame.copy()
            
    cap.release()

# ==========================================
# 🧠 AI WORKER (เลน GPU - YOLO)
# ==========================================
def ai_worker():
    global results_ai, fps_ai, running
    print("🧠 AI Worker: Warming up Blackwell Engine...")
    
    try:
        # โหลดโมเดลไปที่ CUDA
        model = YOLO(MODEL_PATH, task='segment').to('cuda')
        print("✅ YOLO Engine Ready on CUDA!")
    except Exception as e:
        print(f"❌ AI Load Fail: {e}")
        return

    prev_time = time.time()
    while running:
        frame_to_proc = None
        with lock:
            if latest_frame is not None:
                frame_to_proc = latest_frame.copy()

        if frame_to_proc is not None:
            # Inference แบบปิด Log หน้าจอ (Verbose=False) เพื่อความเร็ว
            results = model(frame_to_proc, verbose=False, conf=0.5)
            
            with lock:
                results_ai = results[0].boxes if results else []
            
            # คำนวณ AI FPS
            curr_time = time.time()
            diff = curr_time - prev_time
            if diff > 0:
                fps_ai = (fps_ai * 0.9) + (0.1 * (1.0 / diff))
            prev_time = curr_time
        
        # หน่วงเล็กน้อยเพื่อให้ Thread อื่นได้เดินบ้าง
        time.sleep(0.001)

# ==========================================
# 🖥️ MAIN UI LOOP (เลนแสดงผล)
# ==========================================
def main():
    global running
    print("🚀 Starting PillTrack Native GUI...")
    
    # เริ่ม Threads
    t_cam = threading.Thread(target=camera_worker, daemon=True)
    t_ai = threading.Thread(target=ai_worker, daemon=True)
    t_cam.start()
    t_ai.start()

    cv2.namedWindow("PillTrack AI - Blackwell Edition", cv2.WINDOW_NORMAL)

    while True:
        with lock:
            if latest_frame is None:
                continue
            display = latest_frame.copy()
            current_boxes = results_ai

        # 🎨 วาด UI Overlay
        # 1. Background Bar
        cv2.rectangle(display, (0, 0), (WIDTH, 70), (0, 0, 0), -1)
        
        # 2. FPS Stats
        cv2.putText(display, f"CAM: {int(fps_camera)} FPS", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, f"AI:  {int(fps_ai)} FPS (RTX 5060 Ti)", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 3. วาด Boxes
        if current_boxes is not None:
            for box in current_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "BOX", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # แสดงผล
        cv2.imshow("PillTrack AI - Blackwell Edition", display)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cv2.destroyAllWindows()
    print("👋 System Closed.")

if __name__ == "__main__":
    main()