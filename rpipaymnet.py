import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import socket
import struct
import time
from twilio.rest import Client

# ---------------------------
# CONFIG
# ---------------------------
WINDOWS_IP = "192.168.0.101"   # ?? CHANGE THIS
PORT = 9999

# ---------------------------
# TWILIO CONFIG
# ---------------------------
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # ?? CHANGE THIS
TWILIO_AUTH_TOKEN  = "your_auth_token_here"                # ?? CHANGE THIS
TWILIO_FROM_NUMBER = "+1XXXXXXXXXX"                        # ?? Your Twilio number
USER_PHONE_NUMBER  = "+91XXXXXXXXXX"                       # ?? Receiver's number (with country code)

# ---------------------------
# PAYMENT CONFIG
# ---------------------------
PAYMENT_AMOUNT = 1          # Rs 1
MERCHANT_UPI   = "yourname@okaxis"  # ?? CHANGE THIS to your UPI ID
MERCHANT_NAME  = "Parking System"

def generate_gpay_link(amount, upi_id, name, note="Parking Fee"):
    """Generate a UPI deep link compatible with GPay, PhonePe, Paytm etc."""
    import urllib.parse
    upi_url = (
        f"upi://pay?pa={upi_id}"
        f"&pn={urllib.parse.quote(name)}"
        f"&am={amount}"
        f"&cu=INR"
        f"&tn={urllib.parse.quote(note)}"
    )
    return upi_url

def send_sms_with_payment(plate_text, phone_number):
    """Send SMS with plate info and GPay payment link via Twilio."""
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        gpay_link = generate_gpay_link(
            amount=PAYMENT_AMOUNT,
            upi_id=MERCHANT_UPI,
            name=MERCHANT_NAME,
            note=f"Parking fee for {plate_text}"
        )

        message_body = (
            f"?? Parking Fee Alert!\n"
            f"Vehicle Plate: {plate_text}\n"
            f"Amount Due: Rs {PAYMENT_AMOUNT}\n"
            f"Pay via GPay/UPI:\n{gpay_link}\n"
            f"Thank you!"
        )

        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_FROM_NUMBER,
            to=phone_number
        )

        print(f"? SMS Sent! SID: {message.sid}")
        return True

    except Exception as e:
        print(f"? Twilio Error: {e}")
        return False

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
        ids      = results[0].boxes.id.cpu().numpy().astype(int)
        boxes    = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, class_ids):
            if track_id in processed_ids:
                continue

            x1, y1, x2, y2 = box
            class_name = names[class_id]

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f"{class_name} ID:{track_id}", (x1, y1), 1, 1)

            if class_name.lower() == "licence":
                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue
                if crop.shape[1] < 100 or crop.shape[0] < 30:
                    continue

                # Encode and send image to Windows OCR server
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

                    plate_text = result_data.decode("utf-8").strip()
                    print("Plate from Windows:", plate_text)

                    if plate_text:
                        cvzone.putTextRect(frame, plate_text, (x1, y2 + 30), 1, 2)

                        # ✅ Send SMS with GPay payment link
                        print(f"?? Sending payment SMS for plate: {plate_text}")
                        send_sms_with_payment(plate_text, USER_PHONE_NUMBER)

                    processed_ids.add(track_id)
                    client.close()

                except Exception as e:
                    print("Connection Error:", e)

    cv2.imshow("License Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
picam2.stop()
