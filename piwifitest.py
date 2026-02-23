import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import socket
import struct
import time
import sqlite3
import os
import urllib.parse
import qrcode
import numpy as np
from datetime import datetime
from twilio.rest import Client

# ==============================================================
# CONFIG — EDIT THESE
# ==============================================================
WINDOWS_IP      = "192.168.0.101"    # Your Windows PC IP
PORT            = 9999

# UPI / GPay details (money goes to YOU)
UPI_ID          = "yourname@okicici"  # e.g. 9876543210@ybl
UPI_NAME        = "ParkingSystem"
PARKING_AMOUNT  = 50                  # Amount in INR

# Twilio SMS credentials
TWILIO_SID      = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH     = "your_auth_token"
TWILIO_FROM     = "+1XXXXXXXXXX"

# Set True to send WhatsApp instead of SMS
USE_WHATSAPP    = False
TWILIO_WA_FROM  = "whatsapp:+14155238886"

# SQLite DB on Pi
DB_FILE         = "pi_payment_log.db"
# ==============================================================


# ==============================================================
# SQLite — Pi side (stores payment history)
# ==============================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id     INTEGER,
            plate_number TEXT DEFAULT '',
            phone        TEXT DEFAULT '',
            amount       REAL DEFAULT 0.0,
            upi_link     TEXT DEFAULT '',
            sms_status   TEXT DEFAULT 'not_sent',
            detected_at  TEXT DEFAULT '',
            sent_at      TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Pi database ready: {os.path.abspath(DB_FILE)}")


def log_payment_db(track_id, plate_number, phone, amount, upi_link, sms_status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO payments (track_id, plate_number, phone, amount, upi_link, sms_status, detected_at, sent_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        track_id,
        plate_number,
        phone,
        amount,
        upi_link,
        sms_status,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    print(f"[DB] Payment logged for plate: {plate_number}")


def get_all_payments():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT * FROM payments ORDER BY detected_at DESC").fetchall()
    conn.close()
    return rows


# ==============================================================
# Phone Lookup — SQLite database of plate → phone
# ==============================================================
PLATE_DB_FILE = "plates.db"

def init_plate_db():
    conn = sqlite3.connect(PLATE_DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            plate      TEXT NOT NULL UNIQUE,
            phone      TEXT NOT NULL,
            owner_name TEXT DEFAULT '',
            vehicle    TEXT DEFAULT ''
        )
    """)
    # Insert some sample data if empty
    count = conn.execute("SELECT COUNT(*) FROM plates").fetchone()[0]
    if count == 0:
        sample = [
            ("MH12AB1234", "+919876543210", "Rajesh Kumar",  "Honda City"),
            ("KA01XY5678", "+919123456789", "Priya Sharma",  "Maruti Swift"),
            ("DL3CAF0001", "+919988776655", "Amit Singh",    "Hyundai i20"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO plates (plate, phone, owner_name, vehicle) VALUES (?,?,?,?)",
            sample
        )
    conn.commit()
    conn.close()


def lookup_plate(plate_text: str) -> dict:
    """Returns {'phone': ..., 'owner_name': ..., 'vehicle': ...} or {}"""
    import re
    cleaned = re.sub(r'[\s\-]', '', plate_text.strip().upper())
    try:
        conn = sqlite3.connect(PLATE_DB_FILE)
        row = conn.execute(
            "SELECT * FROM plates WHERE REPLACE(REPLACE(plate,' ',''),'-','') = ?",
            (cleaned,)
        ).fetchone()
        conn.close()
        if row:
            return {"plate": row[1], "phone": row[2], "owner_name": row[3], "vehicle": row[4]}
    except Exception as e:
        print(f"[DB ERROR] lookup_plate: {e}")
    return {}


# ==============================================================
# UPI / GPay Link Generator
# ==============================================================
def generate_upi_link(plate, amount=PARKING_AMOUNT):
    params = {
        "pa": UPI_ID,
        "pn": UPI_NAME,
        "am": str(amount),
        "cu": "INR",
        "tn": f"Parking Fee - {plate}"
    }
    upi_link  = "upi://pay?" + urllib.parse.urlencode(params)
    gpay_link = "https://gpay.app.goo.gl/pay?" + urllib.parse.urlencode(params)
    return upi_link, gpay_link


# ==============================================================
# QR Code overlay on frame
# ==============================================================
def generate_qr_image(link, size=180):
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(link)
    qr.make(fit=True)
    qr_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_np  = np.array(qr_pil)
    return cv2.resize(qr_np, (size, size))


def overlay_qr(frame, qr_img):
    h, w    = frame.shape[:2]
    qh, qw  = qr_img.shape[:2]
    x_off   = w - qw - 10
    y_off   = h - qh - 10
    frame[y_off:y_off+qh, x_off:x_off+qw] = qr_img
    cv2.rectangle(frame, (x_off-2, y_off-2), (x_off+qw+2, y_off+qh+2), (0,255,255), 2)
    cv2.putText(frame, "Scan to Pay", (x_off, y_off-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    return frame


# ==============================================================
# Send SMS / WhatsApp
# ==============================================================
def send_sms(phone, plate, gpay_link):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg = (
            f"Vehicle {plate} detected.\n"
            f"Pay parking fee Rs.{PARKING_AMOUNT} via GPay:\n"
            f"{gpay_link}"
        )
        client.messages.create(body=msg, from_=TWILIO_FROM, to=phone)
        print(f"[SMS] Sent to {phone}")
        return "sent"
    except Exception as e:
        print(f"[SMS ERROR] {e}")
        return "failed"


def send_whatsapp(phone, plate, gpay_link):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg = (
            f"*Vehicle Detected: {plate}*\n"
            f"Pay Rs.{PARKING_AMOUNT} parking fee:\n"
            f"{gpay_link}"
        )
        client.messages.create(body=msg, from_=TWILIO_WA_FROM, to=f"whatsapp:{phone}")
        print(f"[WhatsApp] Sent to {phone}")
        return "sent"
    except Exception as e:
        print(f"[WhatsApp ERROR] {e}")
        return "failed"


# ==============================================================
# Send plate crop to Windows OCR Server
# ==============================================================
def send_to_ocr_server(crop, track_id):
    try:
        _, img_encoded = cv2.imencode(".jpg", crop)
        img_bytes = img_encoded.tobytes()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10)
        client.connect((WINDOWS_IP, PORT))

        client.sendall(struct.pack(">I", track_id))
        client.sendall(struct.pack(">I", len(img_bytes)))
        client.sendall(img_bytes)

        raw_size    = client.recv(4)
        result_size = struct.unpack(">I", raw_size)[0]

        result_data = b""
        while len(result_data) < result_size:
            packet = client.recv(4096)
            if not packet:
                break
            result_data += packet

        client.close()
        return result_data.decode("utf-8").strip()

    except Exception as e:
        print(f"[OCR SERVER ERROR] {e}")
        return ""


# ==============================================================
# MAIN
# ==============================================================
def main():
    init_db()
    init_plate_db()

    model = YOLO("best.pt")
    names = model.names
    print("Model Classes:", names)

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (800, 600)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    time.sleep(1)

    processed_ids = set()
    qr_display    = {}   # track_id → (qr_img, expire_time)

    print("=== Detection Running. Press ESC to quit ===")

    while True:
        frame   = picam2.capture_array()
        frame   = cv2.flip(frame, -1)
        results = model.track(frame, persist=True)

        if results and results[0].boxes.id is not None:
            ids       = results[0].boxes.id.cpu().numpy().astype(int)
            boxes     = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, ids, class_ids):
                x1, y1, x2, y2 = box
                class_name      = names[class_id]

                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cvzone.putTextRect(frame, f"{class_name} ID:{track_id}", (x1, y1), 1, 1)

                if class_name.lower() != "licence":
                    continue

                # Show QR if already processed and still within 15s window
                if track_id in processed_ids:
                    if track_id in qr_display and time.time() < qr_display[track_id][1]:
                        frame = overlay_qr(frame, qr_display[track_id][0])
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                if crop.shape[1] < 60 or crop.shape[0] < 20:
                    continue

                # ---- Send to Windows for OCR ----
                print(f"[INFO] Sending to OCR server (Track ID: {track_id})...")
                plate_text = send_to_ocr_server(crop, track_id)
                print(f"[INFO] Plate: '{plate_text}'")

                if not plate_text:
                    continue

                cvzone.putTextRect(frame, f"Plate: {plate_text}", (x1, y2+10), 1, 2)

                # ---- Generate UPI / GPay Link ----
                clean_plate       = plate_text.strip().upper()
                upi_link, gpay_link = generate_upi_link(clean_plate)

                # ---- Show QR on screen ----
                qr_img = generate_qr_image(upi_link)
                qr_display[track_id] = (qr_img, time.time() + 15)
                frame = overlay_qr(frame, qr_img)

                # ---- Lookup phone from SQLite plates.db ----
                info   = lookup_plate(clean_plate)
                phone  = info.get("phone", "")
                owner  = info.get("owner_name", "Unknown")

                sms_status = "no_phone"

                if phone:
                    print(f"[INFO] Owner: {owner} | Phone: {phone}")
                    cvzone.putTextRect(frame, f"Owner: {owner}", (x1, y2+40), 1, 1)

                    if USE_WHATSAPP:
                        sms_status = send_whatsapp(phone, clean_plate, gpay_link)
                    else:
                        sms_status = send_sms(phone, clean_plate, gpay_link)
                else:
                    print(f"[WARN] No phone found for plate: {clean_plate}")

                # ---- Log everything to SQLite ----
                log_payment_db(
                    track_id     = int(track_id),
                    plate_number = clean_plate,
                    phone        = phone,
                    amount       = PARKING_AMOUNT,
                    upi_link     = gpay_link,
                    sms_status   = sms_status
                )

                processed_ids.add(track_id)

        # Clean up expired QR entries
        qr_display = {k: v for k, v in qr_display.items() if time.time() < v[1]}

        cv2.imshow("License Detection + Payment", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    picam2.stop()

    # Print payment summary on exit
    print("\n===== PAYMENT LOG SUMMARY =====")
    rows = get_all_payments()
    print(f"{'ID':<5} {'TRACK':<7} {'PLATE':<15} {'PHONE':<16} {'AMT':>6}  {'SMS':<10} {'DETECTED AT'}")
    print("-" * 85)
    for r in rows:
        print(f"{r[0]:<5} {r[1]:<7} {r[2]:<15} {r[3]:<16} Rs{r[4]:<5} {r[6]:<10} {r[7]}")


if __name__ == "__main__":
    main()
