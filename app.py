
from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import socket
import struct
import threading
import time
from datetime import datetime
import json
import sqlite3
import os

app = Flask(__name__)

# ---------------------------
# CONFIG
# ---------------------------
WINDOWS_IP = "192.168.0.101"   # ?? CHANGE THIS
PORT = 9999
DB_PATH = "plates.db"

# ---------------------------
# SQLite Setup
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id  INTEGER,
            plate     TEXT NOT NULL,
            date      TEXT NOT NULL,
            time      TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_plate_to_db(track_id, plate, date, time_str, timestamp):
    """Save plate only if same plate text not seen in last 60 seconds."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Check if this plate was already saved in the last 60 seconds
    c.execute(
        "SELECT COUNT(*) FROM plates WHERE plate = ? AND timestamp >= datetime(?, '-60 seconds')",
        (plate, timestamp)
    )
    count = c.fetchone()[0]
    if count == 0:
        c.execute(
            "INSERT INTO plates (track_id, plate, date, time, timestamp) VALUES (?,?,?,?,?)",
            (track_id, plate, date, time_str, timestamp)
        )
        conn.commit()
        conn.close()
        return True   # was saved
    conn.close()
    return False      # duplicate, skipped

def load_processed_plates():
    """On startup, load recently detected plates from DB to avoid re-saving after restart."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT DISTINCT plate FROM plates WHERE timestamp >= datetime('now', '-5 minutes')")
    rows = {r[0] for r in c.fetchall()}
    conn.close()
    return rows

init_db()
recently_seen_plates = load_processed_plates()  # plates seen in last 5 min before restart
print(f"[DB] Loaded {len(recently_seen_plates)} recently seen plates from DB")

# ---------------------------
# Shared State
# ---------------------------
detected_plates = []          # List of dicts: {id, plate, time, date, crop_b64}
processed_ids = set()
frame_lock = threading.Lock()
plates_lock = threading.Lock()
latest_frame = None

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
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.stride = None
picam2.preview_configuration.main.format = "BGR888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ---------------------------
# Detection Thread
# ---------------------------
def detection_loop():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        results = model.track(frame, persist=True)

        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, ids, class_ids):
                x1, y1, x2, y2 = box
                class_name = names[class_id]
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Draw bounding box on live feed
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cvzone.putTextRect(frame, f"{class_name} ID:{track_id}", (x1, max(0, y1 - 10)), 1, 1)

                if class_name.lower() == "licence" and track_id not in processed_ids:
                    crop = frame[y1:y2, x1:x2]
                    if crop is None or crop.size == 0:
                        continue
                    if crop.shape[1] < 100 or crop.shape[0] < 30:
                        continue

                    _, img_encoded = cv2.imencode(".jpg", crop)
                    img_bytes = img_encoded.tobytes()

                    try:
                        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client.settimeout(5)
                        client.connect((WINDOWS_IP, PORT))
                        client.sendall(struct.pack(">I", track_id))
                        client.sendall(struct.pack(">I", len(img_bytes)))
                        client.sendall(img_bytes)

                        result_size_data = client.recv(4)
                        result_size = struct.unpack(">I", result_size_data)[0]
                        result_data = b""
                        while len(result_data) < result_size:
                            packet = client.recv(4096)
                            if not packet:
                                break
                            result_data += packet

                        plate_text = result_data.decode("utf-8").strip()
                        client.close()

                        # Mark track_id as processed regardless (avoid re-sending to OCR)
                        processed_ids.add(track_id)

                        if plate_text and plate_text not in recently_seen_plates:
                            now = datetime.now()
                            saved = save_plate_to_db(
                                int(track_id),
                                plate_text,
                                now.strftime("%d %b %Y"),
                                now.strftime("%H:%M:%S"),
                                now.isoformat()
                            )
                            if saved:
                                recently_seen_plates.add(plate_text)
                                with plates_lock:
                                    detected_plates.insert(0, {
                                        "id": int(track_id),
                                        "plate": plate_text,
                                        "time": now.strftime("%H:%M:%S"),
                                        "date": now.strftime("%d %b %Y"),
                                        "timestamp": now.isoformat()
                                    })
                                    if len(detected_plates) > 50:
                                        detected_plates.pop()
                                print(f"[SAVED] Plate: {plate_text} | ID: {track_id}")
                            else:
                                print(f"[SKIPPED] Duplicate plate: {plate_text}")
                        elif plate_text in recently_seen_plates:
                            print(f"[SKIPPED] Recently seen plate: {plate_text}")

                    except Exception as e:
                        print(f"[ERROR] Connection: {e}")

        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)

# Start detection in background
t = threading.Thread(target=detection_loop, daemon=True)
t.start()

# ---------------------------
# Video Stream Generator
# ---------------------------
def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        # Ensure exactly 640x480
        frame = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/plates')
def get_plates():
    with plates_lock:
        return jsonify(detected_plates[:20])

@app.route('/api/history')
def get_history():
    """Return all saved plates from SQLite DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM plates ORDER BY id DESC LIMIT 200")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM plates")
    total = c.fetchone()[0]
    conn.close()
    return jsonify({
        "total": total,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%A, %d %B %Y")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
