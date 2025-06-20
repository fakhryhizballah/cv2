import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
import os

# Load file .env
load_dotenv()


# Direktori untuk menyimpan wajah
output_dir = "captured_faces"
os.makedirs(output_dir, exist_ok=True)

# Tempat menyimpan encoding wajah yang sudah tersimpan
known_encodings = []

def face_match_percentage(distance):
    return round((1 - distance) * 100, 2)

def is_new_face(face_encoding, known_encodings, min_match_percent=75):
    if not known_encodings:
        return True

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    match_percentages = [(1 - dist) * 100 for dist in distances]
    max_percent = max(match_percentages)

    return max_percent < min_match_percent  # TRUE kalau wajah belum dikenali

def crop_face_with_padding(frame, top, right, bottom, left, padding=50):
    height, width = frame.shape[:2]
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    return frame[top:bottom, left:right]

# Inisialisasi webcam
# video_capture = cv2.VideoCapture(0)
rtsp_url = os.getenv("RSTP_URL")
video_capture = cv2.VideoCapture(rtsp_url)
print("Mulai mendeteksi wajah...")

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1].copy().astype('uint8')

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if is_new_face(face_encoding, known_encodings):
            # Tambahkan ke daftar known
            known_encodings.append(face_encoding)

            # Simpan foto
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
            face_image = crop_face_with_padding(frame, top, right, bottom, left, padding=50)
            cv2.imwrite(filename, face_image)

            print(f"[+] Wajah baru disimpan: {filename}")
        else:
            print("[-] Wajah sudah dikenali, tidak disimpan ulang")

    # Tampilkan hasil di layar
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()