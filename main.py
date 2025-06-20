import face_recognition
import cv2
import pickle

import numpy as np

def face_match_percentage(distance):
    # Konversi jarak ke skor 0-100% (pakai skala buatan yang umum)
    return round((1 - distance) * 100, 2)

# Load encoding wajah
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_ids = pickle.load(f)

cap = cv2.VideoCapture(0)

print("Menjalankan pengenalan wajah...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize untuk percepatan
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_small = small_frame[:, :, ::-1]
    rgb_small = small_frame[:, :, ::-1].copy().astype('uint8')

    face_locations = face_recognition.face_locations(rgb_small)
    
    print(f"Jumlah wajah terdeteksi: {len(face_locations)}")
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        # face_encodings = face_recognition.face_encodings(rgb_small)
    else:
        face_encodings = []
    # face_encodings = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)


        if True in matches:
            index = matches.index(True)
            name = known_ids[index]
            # name = known_names[best_match_index]
            match_score = face_match_percentage(distances[best_match_index])
            print(f"Kecocokan: {name} - {match_score}%")

        # Kembalikan ke ukuran normal
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Gambar kotak & nama
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"{match_score}%", (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Pengenalan Wajah", frame)
    # print(f"Wajah terdeteksi: {name}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()