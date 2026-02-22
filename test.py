import socket
import struct
import numpy as np
import cv2
import os
from datetime import datetime
from paddleocr import PaddleOCR
import re

# ---------------------------
# CONFIG
# ---------------------------
PORT = 9999
SAVE_FOLDER = "received_plates"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ---------------------------
# Load PaddleOCR (once)
# ---------------------------
print("🔄 Loading PaddleOCR...")
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en"
)
print("✅ PaddleOCR Ready!\n")

# ---------------------------
# OCR Function (predict API)
# ---------------------------
def run_ocr(img):
    result = ocr.predict(img)

    plate_text = ""

    if result and isinstance(result, list):
        if len(result) > 0 and "rec_texts" in result[0]:
            texts = result[0]["rec_texts"]
            scores = result[0].get("rec_scores", [])

            for text, score in zip(texts, scores):
                if score > 0.3:
                    plate_text += text.upper().strip() + " "

    # Clean plate
    plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)

    return plate_text.strip()


# ---------------------------
# Start TCP Server
# ---------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(5)

print(f"🚀 Windows OCR Server listening on port {PORT}")
print("Waiting for Raspberry Pi...\n")

while True:
    try:
        conn, addr = server.accept()
        print(f"📡 Connected from: {addr}")

        # ---- Receive Track ID ----
        track_id_data = conn.recv(4)
        track_id = struct.unpack(">I", track_id_data)[0]

        # ---- Receive Image Size ----
        size_data = conn.recv(4)
        size = struct.unpack(">I", size_data)[0]

        print(f"   Track ID: {track_id} | Image Size: {size} bytes")

        # ---- Receive Image Data ----
        data = b""
        while len(data) < size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        # ---- Decode Image ----
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Image decode failed")
            result_bytes = "".encode("utf-8")
            conn.sendall(struct.pack(">I", len(result_bytes)))
            conn.sendall(result_bytes)
            conn.close()
            continue

        # ---- Save Image ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"plate_{track_id}_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        print(f"💾 Saved: {filename}")

        # ---- Run OCR ----
        plate_text = run_ocr(img)
        print(f"🔤 OCR Result: '{plate_text}'")

        # ---- Send Back Result ----
        result_bytes = plate_text.encode("utf-8")
        conn.sendall(struct.pack(">I", len(result_bytes)))
        conn.sendall(result_bytes)

        conn.close()
        print("✅ Result sent back to Pi\n")

    except KeyboardInterrupt:
        print("Server stopped.")
        break
    except Exception as e:
        print("❌ Error:", e)