import cv2
import threading
import time
import numpy as np

class WebcamStream:
    def __init__(self, src=0, width=1280, height=720):
        if src is None:
            self.stream = None
            self.grabbed = False
            self.frame = self.create_dummy_frame("NO CAMERA FOUND")
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            if self.stream and self.stream.isOpened():
                (grabbed, frame) = self.stream.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.grabbed = grabbed
                        self.frame = frame
                else:
                    self.grabbed = False
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()

    def create_dummy_frame(self, text):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, text, (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return img