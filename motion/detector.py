import cv2
import numpy as np
import os
import time
import config


def open_camera(source):
    if os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    return cap


class MotionDetector:
    def __init__(self, video_source=0):
        self.cap = open_camera(video_source)
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40)
        self.previous_centroids = []
        self.frame_counter = 0
        self.alert_active_until = 0.0

    def _process(self, frame):
        self.frame_counter += 1
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        human_like = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w + 0.001)

            if config.MIN_ASPECT_RATIO < aspect_ratio < config.MAX_ASPECT_RATIO:
                cx, cy = x + w // 2, y + h // 2
                self.previous_centroids.append((cx, cy))

                if len(self.previous_centroids) > config.CENTROID_HISTORY:
                    self.previous_centroids.pop(0)

                if len(self.previous_centroids) >= 2:
                    dx = self.previous_centroids[-1][0] - self.previous_centroids[0][0]
                    dy = self.previous_centroids[-1][1] - self.previous_centroids[0][1]
                    distance = np.sqrt(dx**2 + dy**2)
                    avg_speed = distance / len(self.previous_centroids)
                else:
                    avg_speed = 0

                if config.MIN_SPEED < avg_speed < config.MAX_SPEED:
                    human_like = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Human-like Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if human_like:
            self.alert_active_until = time.time() + config.ALERT_DURATION
            cv2.putText(frame, "ALERT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame, fgmask, human_like

    def next_frame(self):
        if not self.cap or not self.cap.isOpened():
            return self._placeholder_jpeg(), None, False

        ret, frame = self.cap.read()
        if not ret:
            return self._placeholder_jpeg(), None, False

        processed, _, human_like = self._process(frame)
        ok, buf = cv2.imencode('.jpg', processed)

        if not ok:
            return self._placeholder_jpeg(), None, False

        return buf.tobytes(), processed, human_like

    def release(self):
        if self.cap:
            self.cap.release()

    def _placeholder_jpeg(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, "NO VIDEO SOURCE", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        ok, buf = cv2.imencode('.jpg', img)
        return buf.tobytes() if ok else None

    def is_alert_active(self):
        return time.time() < self.alert_active_until