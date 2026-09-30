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


def draw_diagonal_gradient(canvas, color_top_left, color_bottom_right):
    """Gradient diagonal halus dari pojok kiri-atas ke kanan-bawah - terasa lebih dinamis
    dibanding gradient vertikal polos."""
    h, w = canvas.shape[:2]
    c1 = np.array(color_top_left, dtype=np.float32)
    c2 = np.array(color_bottom_right, dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    t = ((xx / max(w - 1, 1)) + (yy / max(h - 1, 1))) / 2.0
    t = t[:, :, None]
    grad = (c1[None, None, :] * (1 - t) + c2[None, None, :] * t).astype(np.uint8)
    canvas[:] = grad
    return canvas


def draw_blob(canvas, center, base_radius, color, alpha=0.45, wobble=0.22, seed=0):
    """Menggambar blob organik (bentuk awan/gelembung tidak beraturan) - lebih soft
    dan playful dibanding lingkaran sempurna."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    n_points = 10
    pts = []
    for i in range(n_points):
        angle = 2 * math.pi * i / n_points
        r = base_radius * (1 + rng.uniform(-wobble, wobble))
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        pts.append([x, y])
    pts = np.array([pts], dtype=np.int32)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, pts, color, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def draw_star(canvas, center, size, color, thickness=-1):
    """Menggambar bintang kecil sebagai dekorasi (outline atau solid)."""
    cx, cy = center
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = size if i % 2 == 0 else size * 0.42
        x = int(cx + r * math.cos(angle))
        y = int(cy - r * math.sin(angle))
        pts.append([x, y])
    pts = np.array([pts], dtype=np.int32)
    if thickness == -1:
        cv2.fillPoly(canvas, pts, color, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, pts, True, color, thickness, cv2.LINE_AA)


def scatter_glitter(canvas, count, color, seed=1, size_range=(1, 3)):
    """Menabur titik-titik kecil mengilap (glitter dots) secara acak namun konsisten
    (seeded) di seluruh canvas untuk tekstur lebih hidup."""
    rng = np.random.default_rng(seed)
    h, w = canvas.shape[:2]
    for _ in range(count):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        r = rng.integers(size_range[0], size_range[1] + 1)
        cv2.circle(canvas, (x, y), int(r), color, -1, cv2.LINE_AA)


def draw_drop_shadow(canvas, x, y, w, h, radius, blur_size=25, offset=(0, 10), strength=0.35):
    """Menggambar drop shadow lembut di bawah sebuah rounded-rect area, memberi efek
    'mengambang' (floating card) seperti UI modern."""
    shadow_layer = np.zeros_like(canvas)
    sx, sy = x + offset[0], y + offset[1]
    mask = round_corner_mask(w, h, radius)
    shadow_layer[sy:sy + h, sx:sx + w] = cv2.merge([mask, mask, mask])
    shadow_layer = cv2.GaussianBlur(shadow_layer, (blur_size, blur_size), 0)
    shadow_color = np.array([120, 90, 140], dtype=np.uint8)
    shadow_colored = np.where(shadow_layer > 10, shadow_color, 0).astype(np.uint8)
    alpha_mask = (shadow_layer.astype(np.float32) / 255.0) * strength
    canvas[:] = (canvas.astype(np.float32) * (1 - alpha_mask) +
                 shadow_colored.astype(np.float32) * alpha_mask).astype(np.uint8)


def build_static_template(cam_w, cam_h, is_frozen):

    top_bar = 86
    bottom_bar = 76
    side_margin = 34
    border_thickness = 7
    corner_radius = 30

    canvas_w = cam_w + side_margin * 2
    canvas_h = cam_h + top_bar + bottom_bar

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # ---------- Background: gradient diagonal lembut ----------
    if is_frozen:
        draw_diagonal_gradient(canvas, (245, 220, 205), (235, 200, 255))  # peach -> lavender
    else:
        draw_diagonal_gradient(canvas, (235, 215, 255), (215, 235, 255))  # lavender -> baby blue

    # ---------- Blob organik dekoratif ----------
    draw_blob(canvas, (50, 35), 50, (255, 225, 250), alpha=0.5, seed=2)
    draw_blob(canvas, (canvas_w - 45, canvas_h - 40), 60, (255, 240, 215), alpha=0.45, seed=5)
    draw_blob(canvas, (canvas_w - 55, 55), 26, (220, 255, 245), alpha=0.55, seed=8)
    draw_blob(canvas, (40, canvas_h - 35), 22, (255, 235, 235), alpha=0.5, seed=11)

    # ---------- Glitter dots ----------
    scatter_glitter(canvas, count=22, color=(255, 255, 255), seed=3, size_range=(1, 2))

    # ---------- Bintang dekoratif (outline + solid kombinasi) ----------
    draw_star(canvas, (canvas_w - 60, 32), 9, (255, 255, 255), thickness=-1)
    draw_star(canvas, (32, canvas_h - 28), 7, (255, 255, 255), thickness=-1)
    draw_star(canvas, (canvas_w - 24, top_bar + 14), 6, (255, 255, 255), thickness=2)

    # ---------- Drop shadow di bawah area kamera (efek floating card) ----------
    cam_x, cam_y = side_margin, top_bar
    draw_drop_shadow(canvas, cam_x, cam_y, cam_w, cam_h, corner_radius,
                      blur_size=35, offset=(0, 14), strength=0.30)

    # ---------- Judul atas ----------
    title_text = "~ Foto Kita Blur ~"
    font = cv2.FONT_HERSHEY_DUPLEX
    title_scale = 0.95
    title_thickness = 2
    (tw, th), _ = cv2.getTextSize(title_text, font, title_scale, title_thickness)
    title_x = (canvas_w - tw) // 2
    title_y = (top_bar + th) // 2
    cv2.putText(canvas, title_text, (title_x + 2, title_y + 2), font, title_scale,
                (255, 255, 255), title_thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (title_x, title_y), font, title_scale,
                (150, 95, 200), title_thickness, cv2.LINE_AA)

    # ---------- Outer border pastel ----------
    outer_color = (255, 235, 250) if not is_frozen else (210, 230, 255)
    cv2.rectangle(
        canvas,
        (cam_x - border_thickness, cam_y - border_thickness),
        (cam_x + cam_w + border_thickness, cam_y + cam_h + border_thickness),
        outer_color, border_thickness, cv2.LINE_AA
    )
    # ---------- Inner glow line (garis tipis terang di dalam border) ----------
    cv2.rectangle(
        canvas,
        (cam_x - 2, cam_y - 2),
        (cam_x + cam_w + 2, cam_y + cam_h + 2),
        (255, 255, 255), 1, cv2.LINE_AA
    )

    layout = {
        "cam_x": cam_x, "cam_y": cam_y, "cam_w": cam_w, "cam_h": cam_h,
        "corner_radius": corner_radius,
        "top_bar": top_bar, "bottom_bar": bottom_bar, "side_margin": side_margin,
        "canvas_w": canvas_w, "canvas_h": canvas_h,
        "rounded_mask": round_corner_mask(cam_w, cam_h, corner_radius),
    }
    return canvas, layout


def render_frame(template, layout, cam_frame, is_frozen):
    """
    Dipanggil TIAP FRAME - hanya menempel frame kamera (dengan rounded corner,
    memakai mask yang sudah di-precompute) ke atas template statis, lalu
    menggambar badge status + jam. Jauh lebih ringan daripada merender ulang
    seluruh background/blob/shadow tiap kali.
    """
    canvas = template.copy()  # copy murah (cuma memcpy, tanpa blur apa pun)

    cam_x, cam_y = layout["cam_x"], layout["cam_y"]
    cam_w, cam_h = layout["cam_w"], layout["cam_h"]
    mask_3ch = cv2.merge([layout["rounded_mask"]] * 3)

    region = canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w]
    rounded_cam = np.where(mask_3ch == 255, cam_frame, region)
    canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = rounded_cam

    # ---------- Badge status berbentuk pill (dengan shadow halus) ----------
    font = cv2.FONT_HERSHEY_DUPLEX
    status_text = "captured (^_^)" if is_frozen else "live now! >w<"
    status_bg = (190, 220, 255) if is_frozen else (200, 245, 210)
    status_fg = (90, 60, 60)

    badge_font_scale = 0.6
    (sw, sh), _ = cv2.getTextSize(status_text, font, badge_font_scale, 2)
    pad_x, pad_y = 18, 10
    badge_w, badge_h = sw + pad_x * 2, sh + pad_y * 2
    badge_x = layout["side_margin"]
    badge_y = cam_y + cam_h + (layout["bottom_bar"] - badge_h) // 2

    badge_overlay = canvas.copy()
    cv2.rectangle(badge_overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h),
                  status_bg, -1, cv2.LINE_AA)
    badge_mask = round_corner_mask(badge_w, badge_h, badge_h // 2)
    region2 = canvas[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w]
    badge_color_patch = badge_overlay[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w]
    badge_mask_3ch = cv2.merge([badge_mask, badge_mask, badge_mask])
    blended = np.where(badge_mask_3ch == 255, badge_color_patch, region2)
    canvas[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w] = blended

    cv2.putText(canvas, status_text, (badge_x + pad_x, badge_y + badge_h - pad_y + 2),
                font, badge_font_scale, status_fg, 2, cv2.LINE_AA)

    # ---------- Jam kecil di sisi kanan bawah ----------
    time_text = datetime.now().strftime("%H:%M:%S")
    (tw2, th2), _ = cv2.getTextSize(time_text, font, 0.55, 1)
    time_x = layout["canvas_w"] - layout["side_margin"] - tw2
    time_y = cam_y + cam_h + (layout["bottom_bar"] + th2) // 2
    cv2.putText(canvas, time_text, (time_x, time_y), font, 0.55,
                (110, 90, 140), 1, cv2.LINE_AA)

    return canvas


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Pastikan kamera tersambung dan tidak dipakai aplikasi lain.")
        return

    blur_strength = 15  # ukuran kernel blur (harus ganjil), semakin besar semakin blur
    hold_delay = 0.5     # detik - gesture harus stabil sekian lama dulu sebelum freeze terjadi

    is_frozen = False        # status: apakah sedang freeze
    frozen_frame = None      # frame yang dibekukan (sudah diblur)
    gesture_start_time = None  # waktu kapan gesture mulai terdeteksi terus-menerus

    # ---------- Pre-render template statis SEKALI SAJA (bukan tiap frame) ----------
    ret, sample_frame = cap.read()
    if not ret:
        print("Gagal membaca frame awal dari kamera.")
        return
    sample_frame = cv2.flip(sample_frame, 1)
    cam_h, cam_w = sample_frame.shape[:2]

    template_live, layout = build_static_template(cam_w, cam_h, is_frozen=False)
    template_frozen, _ = build_static_template(cam_w, cam_h, is_frozen=True)

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

        active_template = template_frozen if is_frozen else template_live
        photobooth_frame = render_frame(active_template, layout, display_frame, is_frozen)
        cv2.imshow("Foto Kita Blur - Tekan 'q' untuk keluar", photobooth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()