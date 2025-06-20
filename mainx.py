import cv2
import face_recognition

rtsp_url = "rtsp://admin:RSUD1987%23@10.11.0.20:554/Streaming/channels/101"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Gagal membuka RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)

    for top, right, bottom, left in face_locations:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("RTSP Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()