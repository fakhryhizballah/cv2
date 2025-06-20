import cv2
import face_recognition
import os

output_folder = "known_faces"
os.makedirs(output_folder, exist_ok=True)

# Masukkan ID atau nama
face_id = input("Masukkan ID atau nama: ").strip()

# Buka kamera
cap = cv2.VideoCapture(0)
print("Arahkan wajah ke kamera. Tekan 's' untuk simpan, 'q' untuk batal.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca kamera")
        break

    # Tampilkan frame
    cv2.imshow("Pendaftaran Wajah", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        filepath = os.path.join(output_folder, f"{face_id}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"Wajah disimpan ke {filepath}")
        break
    elif key == ord('q'):
        print("Dibatalkan.")
        break

cap.release()
cv2.destroyAllWindows()