import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import socket
import struct
import time

# ---------------------------
# CONFIG
# ---------------------------
WINDOWS_IP = "192.168.0.101"   # ?? CHANGE THIS
PORT = 9999

# ---------------------------
# Load YOLO Model
# ---------------------------
model = YOLO("best.pt")
names = model.names
print("Model Classes:", names)

# ---------------------------
# Camera Setup
# ---------------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (600, 400)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

processed_ids = set()

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)

    results = model.track(frame, persist=True)

    if results and results[0].boxes.id is not None:

        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):

            if track_id in processed_ids:
                continue

            x1, y1, x2, y2 = box
            class_name = names[class_id]

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cvzone.putTextRect(frame, f"{class_name} ID:{track_id}", (x1, y1), 1, 1)

            if class_name.lower() == "licence":

                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue

                if crop.shape[1] < 100 or crop.shape[0] < 30:
                    continue

                # Encode image
                _, img_encoded = cv2.imencode(".jpg", crop)
                img_bytes = img_encoded.tobytes()

                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.connect((WINDOWS_IP, PORT))

                    # Send track ID
                    client.sendall(struct.pack(">I", track_id))

                    # Send image size
                    client.sendall(struct.pack(">I", len(img_bytes)))

                    # Send image data
                    client.sendall(img_bytes)

                    # Receive result size
                    result_size_data = client.recv(4)
                    result_size = struct.unpack(">I", result_size_data)[0]

                    # Receive result
                    result_data = b""
                    while len(result_data) < result_size:
                        packet = client.recv(4096)
                        if not packet:
                            break
                        result_data += packet

                    plate_text = result_data.decode("utf-8")

                    print("Plate from Windows:", plate_text)

                    if plate_text:
                        cvzone.putTextRect(frame, plate_text, (x1, y2+30), 1, 2)

                    processed_ids.add(track_id)
                    client.close()

                except Exception as e:
                    print("Connection Error:", e)

    cv2.imshow("License Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
picam2.stop()