"""
╔══════════════════════════════════════════════════════════════╗
║          🐟  Fish Freshness Detection — Real-Time           ║
║          Model  : Ultralytics YOLO (best.pt)                ║
║          Display: OpenCV  |  Pilih kamera saat startup      ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import time
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics tidak ditemukan. Jalankan: pip install ultralytics")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════

MODEL_PATH   = os.getenv("MODEL_PATH", "best.pt")
CONFIDENCE   = float(os.getenv("CONFIDENCE", "0.5"))
FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
MAX_CAMERAS  = 6   # jumlah index kamera yang akan dicek saat scan

WINDOW_TITLE = "Fish Freshness Detection  |  Q: Keluar  S: Screenshot"

# ── Palet Warna (BGR) ─────────────────────────────────────────
C_BG_DARK    = (15,  15,  20)
C_BG_PANEL   = (22,  27,  34)
C_ACCENT     = (0,  210, 140)    # hijau toska — warna utama UI
C_DANGER     = (50,  60, 220)    # merah
C_WARN       = (30, 160, 240)    # oranye
C_TEXT_MAIN  = (220, 230, 240)
C_TEXT_DIM   = (110, 120, 135)
C_WHITE      = (255, 255, 255)

# Warna label (sesuaikan dengan kelas di best.pt Anda)
LABEL_COLORS = {
    "fresh"     : C_ACCENT,
    "segar"     : C_ACCENT,
    "not_fresh" : C_DANGER,
    "tidak_segar": C_DANGER,
    "stale"     : C_DANGER,
    "unknown"   : C_WARN,
}


# ══════════════════════════════════════════════════════════════
#  SCAN KAMERA
# ══════════════════════════════════════════════════════════════

def scan_cameras(max_index: int = MAX_CAMERAS) -> list[dict]:
    """Scan semua kamera yang tersedia dan kembalikan daftarnya."""
    print("\n╔══════════════════════════════════════════╗")
    print("║      🔍  Mencari kamera tersedia...      ║")
    print("╚══════════════════════════════════════════╝\n")

    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": i, "width": w, "height": h})
                print(f"  ✅  [{i}] Kamera ditemukan  —  {w} x {h}")
            cap.release()
        else:
            print(f"  ❌  [{i}] Tidak tersedia")

    print()
    return cameras


def select_camera(cameras: list[dict]) -> int:
    """Tampilkan menu pilih kamera, kembalikan index yang dipilih."""
    if not cameras:
        print("[ERROR] Tidak ada kamera yang ditemukan!")
        sys.exit(1)

    if len(cameras) == 1:
        cam = cameras[0]
        print(f"[INFO] Hanya ada 1 kamera (index {cam['index']}), langsung digunakan.\n")
        return cam["index"]

    print("╔══════════════════════════════════════════╗")
    print("║         🎥  Pilih Kamera                 ║")
    print("╠══════════════════════════════════════════╣")
    for cam in cameras:
        label = "Webcam bawaan" if cam["index"] == 0 else f"Kamera eksternal / DroidCam"
        print(f"║  [{cam['index']}]  {label:<30}  ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        try:
            choice = input("\n  Masukkan nomor kamera: ").strip()
            idx = int(choice)
            if idx in [c["index"] for c in cameras]:
                print(f"\n[INFO] Menggunakan kamera index {idx}\n")
                return idx
            print("  [!] Pilihan tidak valid, coba lagi.")
        except (ValueError, KeyboardInterrupt):
            print("\n[INFO] Menggunakan kamera default (index 0)")
            return cameras[0]["index"]


# ══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════

def get_label_color(label: str) -> tuple:
    label_lower = label.lower()
    for key, color in LABEL_COLORS.items():
        if key in label_lower:
            return color
    return C_ACCENT


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=10, filled=False):
    """Gambar persegi panjang dengan sudut membulat."""
    x1, y1 = pt1
    x2, y2 = pt2
    if filled:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 270, color, thickness)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 270, 360, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 0, 270, 360, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r), 0, 90, 180, color, thickness)


def draw_confidence_bar(img, x, y, width, conf, color):
    """Bar horizontal yang menunjukkan confidence score."""
    bar_h   = 6
    bar_bg  = (50, 55, 65)
    filled  = int(width * conf)
    cv2.rectangle(img, (x, y), (x + width, y + bar_h), bar_bg, -1)
    cv2.rectangle(img, (x, y), (x + filled, y + bar_h), color, -1)


def draw_detections(frame, results) -> tuple:
    """Gambar bounding box & label yang bersih di atas frame."""
    detection_count = 0
    detection_list  = []   # [(label, conf), ...]

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf  = float(box.conf[0])
            if conf < CONFIDENCE:
                continue

            cls   = int(box.cls[0])
            label = result.names.get(cls, f"cls_{cls}")
            color = get_label_color(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_count += 1
            detection_list.append((label, conf, color))

            # ── Bounding box ──────────────────────────────────
            # Outer glow tipis
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2),
                          tuple(max(0, c-80) for c in color), 1)
            # Kotak utama
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Sudut tebal (corner marks)
            corner = 18
            thick  = 3
            for (sx, sy, dx, dy) in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (sx, sy), (sx + dx*corner, sy), color, thick)
                cv2.line(frame, (sx, sy), (sx, sy + dy*corner), color, thick)

            # ── Label badge ───────────────────────────────────
            text      = f"{label.upper()}  {conf:.0%}"
            font      = cv2.FONT_HERSHEY_DUPLEX
            fs        = 0.55
            th        = 1
            (tw, txh), bl = cv2.getTextSize(text, font, fs, th)
            pad       = 7
            bx1, by1  = x1, y1 - txh - pad*2 - 2
            bx2, by2  = x1 + tw + pad*2, y1

            # Badge background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Badge text
            cv2.putText(frame, text,
                        (bx1 + pad, by2 - pad + 1),
                        font, fs, C_BG_DARK, th+1, cv2.LINE_AA)
            cv2.putText(frame, text,
                        (bx1 + pad, by2 - pad + 1),
                        font, fs, C_WHITE, th, cv2.LINE_AA)

    return frame, detection_count, detection_list


def draw_info_panel(frame, fps: float, detection_list: list,
                    camera_index: int, model_name: str) -> any:
    """Panel info di sisi kanan: FPS, deteksi, kelas."""
    h, w = frame.shape[:2]

    PANEL_W = 240
    px      = w - PANEL_W - 10
    py      = 10

    # ── Background panel ─────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 10, py), (w - 5, h - 10), C_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Garis aksen kiri panel
    cv2.rectangle(frame, (px - 10, py), (px - 7, h - 10), C_ACCENT, -1)

    cy = py + 20   # cursor y

    # ── Judul ────────────────────────────────────────────────
    cv2.putText(frame, "FISH DETECTOR", (px, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, C_ACCENT, 1, cv2.LINE_AA)
    cy += 18
    cv2.putText(frame, model_name, (px, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.9, C_TEXT_DIM, 1, cv2.LINE_AA)
    cy += 4
    cv2.line(frame, (px, cy), (w - 15, cy), (40, 45, 55), 1)
    cy += 16

    # ── FPS ──────────────────────────────────────────────────
    fps_color = C_ACCENT if fps >= 20 else C_WARN if fps >= 10 else C_DANGER
    cv2.putText(frame, "FPS", (px, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:5.1f}", (px + 130, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, fps_color, 1, cv2.LINE_AA)
    cy += 22

    # ── Kamera ───────────────────────────────────────────────
    cam_label = "Webcam Bawaan" if camera_index == 0 else f"Kamera #{camera_index}"
    cv2.putText(frame, "KAMERA", (px, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, cam_label, (px + 80, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_MAIN, 1, cv2.LINE_AA)
    cy += 22

    # ── Timestamp ────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, "WAKTU", (px, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, ts, (px + 80, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_MAIN, 1, cv2.LINE_AA)
    cy += 10
    cv2.line(frame, (px, cy), (w - 15, cy), (40, 45, 55), 1)
    cy += 18

    # ── Deteksi ──────────────────────────────────────────────
    total = len(detection_list)
    det_color = C_ACCENT if total > 0 else C_TEXT_DIM
    cv2.putText(frame, "TERDETEKSI", (px, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, str(total), (px + 145, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, det_color, 1, cv2.LINE_AA)
    cy += 20

    # ── Daftar hasil deteksi ─────────────────────────────────
    for label, conf, color in detection_list[:6]:   # max tampil 6
        # Dot warna
        cv2.circle(frame, (px + 6, cy - 5), 5, color, -1)
        # Label
        disp = f"{label[:16]}"
        cv2.putText(frame, disp, (px + 18, cy),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, C_TEXT_MAIN, 1, cv2.LINE_AA)
        # Confidence bar + angka
        bar_x = px + 18
        bar_w = PANEL_W - 90
        draw_confidence_bar(frame, bar_x, cy + 4, bar_w, conf, color)
        cv2.putText(frame, f"{conf:.0%}", (bar_x + bar_w + 6, cy),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1, cv2.LINE_AA)
        cy += 28

    # ── Footer shortcut ──────────────────────────────────────
    cv2.line(frame, (px, h - 52), (w - 15, h - 52), (40, 45, 55), 1)
    cv2.putText(frame, "[Q] Keluar    [S] Screenshot",
                (px, h - 36),
                cv2.FONT_HERSHEY_PLAIN, 0.85, C_TEXT_DIM, 1, cv2.LINE_AA)

    return frame


def draw_top_bar(frame, screenshot_flash: int) -> any:
    """Bar tipis di atas dengan status & nama aplikasi."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Dot hijau animasi (berkedip tiap detik)
    dot_color = C_ACCENT if int(time.time()) % 2 == 0 else (0, 120, 80)
    cv2.circle(frame, (16, 19), 6, dot_color, -1)
    cv2.putText(frame, "LIVE", (26, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, C_ACCENT, 1, cv2.LINE_AA)

    cv2.putText(frame, "Fish Freshness Detection System",
                (75, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, C_TEXT_MAIN, 1, cv2.LINE_AA)

    # Flash "SCREENSHOT SAVED" saat S ditekan
    if screenshot_flash > 0:
        cv2.putText(frame, "📸  SCREENSHOT DISIMPAN",
                    (w - 310, 25), cv2.FONT_HERSHEY_DUPLEX, 0.48, C_ACCENT, 1, cv2.LINE_AA)

    return frame


def save_screenshot(frame) -> str:
    folder = Path("screenshots")
    folder.mkdir(exist_ok=True)
    fname = folder / f"fish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(fname), frame)
    return str(fname)


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════

def run_detection(camera_index: int, model_path: str):
    # ── Load model ────────────────────────────────────────────
    print(f"[INFO] Memuat model: {model_path}")
    if not Path(model_path).exists():
        print(f"[ERROR] File '{model_path}' tidak ditemukan!")
        print("        Letakkan best.pt di folder yang sama dengan main.py")
        sys.exit(1)

    model      = YOLO(model_path)
    model_name = Path(model_path).name
    classes    = list(model.names.values())
    print(f"[INFO] Model dimuat ✅  |  Kelas: {classes}\n")

    # ── Buka kamera ───────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka kamera index {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Kamera berjalan  —  {aw}x{ah}")
    print(f"[INFO] Confidence threshold: {CONFIDENCE:.0%}")
    print(f"[INFO] Tekan Q untuk keluar, S untuk screenshot\n")

    prev_time        = time.time()
    fps              = 0.0
    screenshot_flash = 0   # countdown frame untuk flash notif

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal baca frame, mencoba lagi...")
            time.sleep(0.05)
            continue

        # ── FPS ───────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── Inferensi ─────────────────────────────────────────
        results = model(frame, verbose=False)

        # ── Anotasi frame ─────────────────────────────────────
        frame, det_count, det_list = draw_detections(frame, results)
        frame = draw_top_bar(frame, screenshot_flash)
        frame = draw_info_panel(frame, fps, det_list, camera_index, model_name)

        if screenshot_flash > 0:
            screenshot_flash -= 1

        # ── Tampilkan ─────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        # ── Keyboard ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q atau ESC
            print("\n[INFO] Keluar...")
            break

        elif key in (ord('s'), ord('S')):
            path = save_screenshot(frame)
            screenshot_flash = 90             # tampilkan notif ~3 detik
            print(f"[INFO] Screenshot: {path}")

        # Deteksi window ditutup
        try:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fish Freshness Detection — Real-Time"
    )
    parser.add_argument("--camera", type=int, default=None,
                        help="Langsung pakai kamera index tertentu (skip menu pilih)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"Path ke model YOLO (default: {MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Confidence threshold 0.0-1.0 (default: {CONFIDENCE})")
    args = parser.parse_args()

    global CONFIDENCE
    CONFIDENCE = args.conf

    print("\n╔══════════════════════════════════════════╗")
    print("║   🐟  Fish Freshness Detection System   ║")
    print("╚══════════════════════════════════════════╝")

    # Pilih kamera
    if args.camera is not None:
        camera_index = args.camera
        print(f"\n[INFO] Menggunakan kamera index {camera_index} (dari argumen)\n")
    else:
        cameras      = scan_cameras()
        camera_index = select_camera(cameras)

    run_detection(camera_index=camera_index, model_path=args.model)


if __name__ == "__main__":
    main()