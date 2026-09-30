"""
Foto Kita Blur - Camera Gesture Blur + Freeze (Photobooth Style)
==================================================================
Buka kamera dengan tampilan ala photobooth (border hitam, judul,
strip info), deteksi gesture ✌️ (peace sign / dua jari).
Saat gesture terdeteksi: frame di-blur lalu DIBEKUKAN (freeze) -
seperti efek "jepret foto". Begitu tangan ✌️ tidak terdeteksi lagi,
kamera otomatis kembali live (normal).

Kontrol:
- Tunjukkan gesture ✌️ (peace sign) ke kamera -> blur + freeze
- Lepas gesture (tangan tidak ✌️ lagi) -> kamera lanjut live lagi
- Tekan 'q' untuk keluar

Requirement:
    pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import math
import time
import numpy as np
from datetime import datetime

# ---------- Setup MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Index landmark untuk tiap jari (tip, dan sendi bawahnya untuk cek lurus/tidak)
FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


def jarak(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_finger_extended(landmarks, tip_id, pip_id, wrist):
    """Cek apakah jari terentang (lurus) dengan membandingkan jarak
    ujung jari ke pergelangan tangan vs jarak sendi ke pergelangan tangan."""
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]
    return jarak(tip, wrist) > jarak(pip, wrist)


def detect_peace_sign(hand_landmarks):
    """
    Deteksi gesture ✌️ (peace sign):
    - Jari telunjuk (index) dan tengah (middle) terentang
    - Jari manis (ring) dan kelingking (pinky) terlipat
    - (Jempol diabaikan, boleh terentang atau tidak)
    """
    lm = hand_landmarks.landmark
    wrist = lm[0]

    index_up = is_finger_extended(lm, 8, 6, wrist)
    middle_up = is_finger_extended(lm, 12, 10, wrist)
    ring_up = is_finger_extended(lm, 16, 14, wrist)
    pinky_up = is_finger_extended(lm, 20, 18, wrist)

    # Pastikan index & middle terpisah (bukan cuma 1 jari)
    spread = jarak(lm[8], lm[12]) > jarak(lm[5], lm[9]) * 0.5

    return index_up and middle_up and not ring_up and not pinky_up and spread


def build_photobooth_frame(cam_frame, is_frozen):
    """
    Membungkus frame kamera dengan layout ala photobooth:
    - Background hitam penuh
    - Judul di bagian atas
    - Border tipis putih mengelilingi area kamera
    - Strip info (status + waktu) di bagian bawah
    - Sudut-sudut dekoratif kecil di tiap pojok frame kamera
    """
    cam_h, cam_w = cam_frame.shape[:2]

    top_bar = 70        # tinggi area judul
    bottom_bar = 60      # tinggi area strip info
    side_margin = 24     # margin kiri-kanan
    border_thickness = 3 # tebal garis border kamera

    canvas_w = cam_w + side_margin * 2
    canvas_h = cam_h + top_bar + bottom_bar

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # hitam penuh

    # ---------- Judul atas ----------
    title_text = "FOTO KITA BLUR"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.0
    title_thickness = 2
    (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, title_thickness)
    title_x = (canvas_w - tw) // 2
    title_y = (top_bar + th) // 2
    cv2.putText(canvas, title_text, (title_x, title_y), font, title_scale,
                (255, 255, 255), title_thickness, cv2.LINE_AA)

    # garis tipis pemisah di bawah judul
    cv2.line(canvas, (side_margin, top_bar - 8), (canvas_w - side_margin, top_bar - 8),
              (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- Tempel frame kamera ----------
    cam_x, cam_y = side_margin, top_bar
    canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = cam_frame

    # ---------- Border putih mengelilingi kamera ----------
    cv2.rectangle(
        canvas,
        (cam_x - border_thickness, cam_y - border_thickness),
        (cam_x + cam_w + border_thickness, cam_y + cam_h + border_thickness),
        (255, 255, 255), border_thickness
    )

    # ---------- Sudut dekoratif (corner brackets) ala photobooth ----------
    corner_len = 26
    corner_thick = 3
    corner_color = (255, 255, 255)
    corners = [
        (cam_x, cam_y, 1, 1),                          # kiri atas
        (cam_x + cam_w, cam_y, -1, 1),                  # kanan atas
        (cam_x, cam_y + cam_h, 1, -1),                  # kiri bawah
        (cam_x + cam_w, cam_y + cam_h, -1, -1),         # kanan bawah
    ]
    for cx, cy, dx, dy in corners:
        cv2.line(canvas, (cx, cy), (cx + dx * corner_len, cy), corner_color, corner_thick, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * corner_len), corner_color, corner_thick, cv2.LINE_AA)

    # ---------- Strip info bawah ----------
    status_text = "● CAPTURED" if is_frozen else "● LIVE"
    status_color = (255, 255, 255) if is_frozen else (0, 255, 0)
    time_text = datetime.now().strftime("%H:%M:%S")

    info_y = cam_y + cam_h + 38
    cv2.putText(canvas, status_text, (side_margin + 4, info_y), font, 0.6,
                status_color, 2, cv2.LINE_AA)

    (tw2, _), _ = cv2.getTextSize(time_text, font, 0.6, 2)
    cv2.putText(canvas, time_text, (canvas_w - side_margin - tw2 - 4, info_y), font, 0.6,
                (200, 200, 200), 2, cv2.LINE_AA)

    return canvas


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Pastikan kamera tersambung dan tidak dipakai aplikasi lain.")
        return

    blur_strength = 35  # ukuran kernel blur (harus ganjil), semakin besar semakin blur
    hold_delay = 0.5     # detik - gesture harus stabil sekian lama dulu sebelum freeze terjadi

    is_frozen = False        # status: apakah sedang freeze
    frozen_frame = None      # frame yang dibekukan (sudah diblur)
    gesture_start_time = None  # waktu kapan gesture mulai terdeteksi terus-menerus

    print("Kamera aktif. Tunjukkan gesture ✌️ untuk blur+freeze. Lepas gesture untuk lanjut live. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)  # mirror biar natural seperti cermin
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        peace_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if detect_peace_sign(hand_landmarks):
                    peace_detected = True
                # optional: gambar titik-titik tangan (bisa di-nonaktifkan)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if peace_detected:
            if not is_frozen:
                if gesture_start_time is None:
                    # Gesture baru mulai terdeteksi -> mulai hitung waktu tunda
                    gesture_start_time = time.time()

                elapsed = time.time() - gesture_start_time
                if elapsed >= hold_delay:
                    # Sudah stabil selama hold_delay detik -> ambil frame ini, blur, lalu bekukan
                    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
                    frozen_frame = blurred
                    is_frozen = True
                    display_frame = frozen_frame
                else:
                    # Masih dalam masa tunda -> tetap tampilkan live dulu
                    display_frame = frame
            else:
                # Selama gesture masih terdeteksi, tetap tampilkan frame yang dibekukan
                display_frame = frozen_frame
        else:
            # Gesture tidak terdeteksi -> reset semua, lanjut live lagi
            is_frozen = False
            frozen_frame = None
            gesture_start_time = None
            display_frame = frame

        photobooth_frame = build_photobooth_frame(display_frame, is_frozen)
        cv2.imshow("Foto Kita Blur - Tekan 'q' untuk keluar", photobooth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()