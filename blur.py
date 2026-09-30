"""
Foto Kita Blur - Camera Gesture Blur
=====================================
Buka kamera, deteksi gesture ✌️ (peace sign / dua jari),
dan blur tampilan kamera selama gesture itu terdeteksi.

Kontrol:
- Tunjukkan gesture ✌️ (peace sign) ke kamera -> layar jadi blur
- Tekan 'q' untuk keluar

Requirement:
    pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import math

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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Tidak bisa membuka kamera. Pastikan kamera tersambung dan tidak dipakai aplikasi lain.")
        return

    blur_strength = 35  # ukuran kernel blur (harus ganjil), semakin besar semakin blur

    print("Kamera aktif. Tunjukkan gesture ✌️ untuk blur. Tekan 'q' untuk keluar.")

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
            frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
            cv2.putText(
                frame, "...", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

        cv2.imshow("Foto Kita Blur - Tekan 'q' untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()