import face_recognition
import os
import pickle

known_encodings = []
known_ids = []

folder = "known_faces"

for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            id_name = os.path.splitext(filename)[0]
            known_ids.append(id_name)
            print(f"Tersimpan: {id_name}")
        else:
            print(f"❌ Tidak ada wajah terdeteksi di {filename}")

# Simpan ke file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_ids), f)