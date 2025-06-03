from ultralytics import YOLO
import cv2

# Muat model YOLOv8 (pastikan file yolov8n.pt berada di direktori yang sama)
model = YOLO('fishgen.pt')

# Buka kamera (0 biasanya untuk kamera bawaan laptop)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Jalankan deteksi objek
    results = model(frame, stream=True)

    # Tampilkan hasil deteksi dengan bounding box dan label
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            conf = float(box.conf[0])               # Confidence
            cls = int(box.cls[0])                   # Kelas
            label = model.names[cls]                # Nama kelas

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
