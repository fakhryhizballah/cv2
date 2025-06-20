import cv2
import face_recognition
import threading
import os
from datetime import datetime
import numpy as np

# Direktori simpan
output_dir = "captured_faces"
os.makedirs(output_dir, exist_ok=True)

known_encodings = []

def face_match_percentage(distance):
    return round((1 - distance) * 100, 2)

def is_new_face(face_encoding, threshold_percent=75):
    if not known_encodings:
        return True
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    max_percent = max((1 - d) * 100 for d in distances)
    return max_percent < threshold_percent

def crop_face_with_padding(frame, top, right, bottom, left, padding=50):
    h, w = frame.shape[:2]
    top = max(0, top - padding)
    right = min(w, right + padding)
    bottom = min(h, bottom + padding)
    left = max(0, left - padding)
    return frame[top:bottom, left:right]

def process_camera(source, cam_name="cam0"):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{cam_name}] Frame gagal dibaca.")
            continue

        rgb = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            if is_new_face(encoding):
                known_encodings.append(encoding)

                cropped = crop_face_with_padding(frame, top, right, bottom, left)
                filename = os.path.join(output_dir, f"{cam_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, cropped)
                print(f"[{cam_name}] [+] Wajah baru disimpan: {filename}")
            else:
                print(f"[{cam_name}] [-] Wajah sudah dikenali, tidak disimpan ulang")

        cv2.imshow(f"{cam_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Jalankan dua kamera secara paralel
camera_sources = ["rtsp://admin:RSUD1987%23@10.11.0.21:554/Streaming/channels/102", "rtsp://admin:RSUD1987%23@10.11.0.20:554/Streaming/channels/102","rtsp://admin:RSUD1987%23@10.11.0.24:554/Streaming/channels/102"]  # atau ganti dengan URL RTSP: ["rtsp://cam1", "rtsp://cam2"]
threads = []

for i, source in enumerate(camera_sources):
    t = threading.Thread(target=process_camera, args=(source, f"cam{i}"))
    t.start()
    threads.append(t)

for t in threads:
    t.join()