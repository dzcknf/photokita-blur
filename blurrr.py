"""
Foto Kita Blur - Camera Gesture Blur + Freeze (Pastel Kawaii Style)
=====================================================================
Buka kamera dengan tampilan pastel kawaii (gradient lembut, rounded
corner, badge lucu), deteksi gesture ✌️ (peace sign / dua jari).
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


def round_corner_mask(width, height, radius):
    """Membuat mask hitam-putih untuk rounded corner (dipakai untuk crop sudut membulat)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)
    return mask


def apply_rounded_corners(image, radius, bg_color):
    """Menerapkan rounded corner pada sebuah image, area luar diisi bg_color."""
    h, w = image.shape[:2]
    mask = round_corner_mask(w, h, radius)
    bg = np.full_like(image, bg_color, dtype=np.uint8)
    mask_3ch = cv2.merge([mask, mask, mask])
    rounded = np.where(mask_3ch == 255, image, bg)
    return rounded.astype(np.uint8)


def draw_pastel_gradient(canvas):
    """Mengisi canvas dengan gradient pastel lembut pink -> lavender (vertikal)."""
    h, w = canvas.shape[:2]
    top_color = np.array([235, 200, 255])     # lavender muda (BGR)
    bottom_color = np.array([225, 210, 255])  # pink pastel (BGR)
    for y in range(h):
        t = y / max(h - 1, 1)
        color = (top_color * (1 - t) + bottom_color * t).astype(np.uint8)
        canvas[y, :] = color
    return canvas


def draw_soft_circle(canvas, center, radius, color, alpha=0.5):
    """Menggambar lingkaran lembut (semi-transparan) sebagai dekorasi kawaii."""
    overlay = canvas.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def draw_star(canvas, center, size, color):
    """Menggambar bintang kecil sederhana sebagai dekorasi."""
    cx, cy = center
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = size if i % 2 == 0 else size * 0.45
        x = int(cx + r * math.cos(angle))
        y = int(cy - r * math.sin(angle))
        pts.append([x, y])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(canvas, pts, color, cv2.LINE_AA)


def build_photobooth_frame(cam_frame, is_frozen):
    """
    Layout pastel kawaii:
    - Background gradient pastel lembut (lavender -> pink)
    - Lingkaran-lingkaran dekoratif transparan di belakang
    - Bintang kecil di pojok atas
    - Frame kamera dengan rounded corner + border tebal pastel
    - Judul dengan font bulat dan warna lembut
    - Badge status berbentuk pill (rounded) di bawah
    """
    cam_h, cam_w = cam_frame.shape[:2]

    top_bar = 80
    bottom_bar = 70
    side_margin = 30
    border_thickness = 8     # border tebal ala polaroid pastel
    corner_radius = 28       # radius rounded corner kamera

    canvas_w = cam_w + side_margin * 2
    canvas_h = cam_h + top_bar + bottom_bar

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    draw_pastel_gradient(canvas)

    # ---------- Dekorasi lingkaran transparan ----------
    draw_soft_circle(canvas, (40, 30), 45, (255, 220, 245), 0.4)
    draw_soft_circle(canvas, (canvas_w - 35, canvas_h - 30), 55, (255, 235, 210), 0.4)
    draw_soft_circle(canvas, (canvas_w - 50, 50), 20, (220, 255, 245), 0.5)

    # ---------- Bintang dekoratif ----------
    draw_star(canvas, (canvas_w - 55, 30), 10, (255, 255, 255))
    draw_star(canvas, (30, canvas_h - 25), 8, (255, 255, 255))

    # ---------- Judul atas ----------
    title_text = "~ Foto Kita Blur ~"
    font = cv2.FONT_HERSHEY_DUPLEX
    title_scale = 0.95
    title_thickness = 2
    (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, title_thickness)
    title_x = (canvas_w - tw) // 2
    title_y = (top_bar + th) // 2
    # bayangan tipis di bawah teks biar lebih "pop"
    cv2.putText(canvas, title_text, (title_x + 2, title_y + 2), font, title_scale,
                (255, 255, 255), title_thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (title_x, title_y), font, title_scale,
                (130, 90, 200), title_thickness, cv2.LINE_AA)

    # ---------- Tempel frame kamera dengan rounded corner ----------
    cam_x, cam_y = side_margin, top_bar
    bg_sample = tuple(int(c) for c in canvas[cam_y, cam_x])
    rounded_cam = apply_rounded_corners(cam_frame, corner_radius, bg_sample)
    canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = rounded_cam

    # ---------- Border pastel tebal mengelilingi kamera (rounded) ----------
    border_color = (235, 200, 255) if not is_frozen else (190, 220, 255)
    cv2.rectangle(
        canvas,
        (cam_x - border_thickness, cam_y - border_thickness),
        (cam_x + cam_w + border_thickness, cam_y + cam_h + border_thickness),
        border_color, border_thickness, cv2.LINE_AA
    )

    # ---------- Badge status berbentuk pill ----------
    status_text = "captured (^_^)" if is_frozen else "live now!"
    status_bg = (190, 220, 255) if is_frozen else (200, 245, 210)
    status_fg = (90, 60, 60)

    badge_font_scale = 0.6
    (sw, sh), _ = cv2.getTextSize(status_text, font, badge_font_scale, 2)
    pad_x, pad_y = 16, 10
    badge_w, badge_h = sw + pad_x * 2, sh + pad_y * 2
    badge_x = side_margin
    badge_y = cam_y + cam_h + (bottom_bar - badge_h) // 2

    badge_overlay = canvas.copy()
    cv2.rectangle(badge_overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                  status_bg, -1, cv2.LINE_AA)
    badge_mask = round_corner_mask(badge_w, badge_h, badge_h // 2)
    region = canvas[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w]
    badge_color_patch = badge_overlay[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w]
    mask_3ch = cv2.merge([badge_mask, badge_mask, badge_mask])
    blended = np.where(mask_3ch == 255, badge_color_patch, region)
    canvas[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w] = blended

    cv2.putText(canvas, status_text, (badge_x + pad_x, badge_y + badge_h - pad_y + 2),
                font, badge_font_scale, status_fg, 2, cv2.LINE_AA)

    # ---------- Jam kecil di sisi kanan bawah ----------
    time_text = datetime.now().strftime("%H:%M:%S")
    (tw2, th2), _ = cv2.getTextSize(time_text, font, 0.55, 1)
    time_x = canvas_w - side_margin - tw2
    time_y = cam_y + cam_h + (bottom_bar + th2) // 2
    cv2.putText(canvas, time_text, (time_x, time_y), font, 0.55,
                (110, 90, 140), 1, cv2.LINE_AA)

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