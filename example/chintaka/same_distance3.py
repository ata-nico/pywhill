#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import math

# --- 設定 ---
CAMERA_ID       = 0               # カメラID
ROI_ANGLE_DEG   = 60              # 前方視野中心±ROI角度 [deg]
FOV_DEG         = 360             # パノラマ水平視野角 [deg]
CONF_THRESH     = 0.2             # YOLO信頼度閾値
RSS_THRESH      = 2500            # RSS閾値
SIDE_THRESH_DEG = 10              # 真横検出閾値 [deg]
CAM_H           = 1.55            # カメラ高さ [m]
YOLO_INPUT_SIZE = (640, 480)      # リサイズ入力サイズ
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"

# --- ウィンドウ ---
WINDOW_NAME = 'Real-Time RSS Detection'

# --- モデルロード ---
model = YOLO(MODEL_PATH)

# --- カメラ ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"Camera {CAMERA_ID} could not be opened.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    sw, sh = YOLO_INPUT_SIZE
    scale_x = w / sw
    scale_y = h / sh

    # YOLO 推論
    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    all_dists = []
    roi_dists = []
    side_dists = []

    for x1, y1, x2, y2 in boxes:
        # 小画像座標から実画像座標へ変換
        rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
        rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # 角度 (小画像基準)
        theta = ((cx / sw) - 0.5) * FOV_DEG
        # 距離計算（実画像高さを使う）
        y_max = max(cy, y1, y2)
        ry_max = y_max * scale_y
        phi = ((ry_max / h) - 0.5) * math.pi
        Dg = float('inf') if abs(phi) < 1e-3 else CAM_H / math.tan(abs(phi))

        all_dists.append((rx1, ry1, rx2, ry2, Dg))
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_dists.append((rx1, ry1, rx2, ry2, Dg))
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            side_dists.append((rx1, ry1, rx2, ry2, Dg))

    # ROI枠 (白)
    for x1, y1, x2, y2, _ in roi_dists:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    # 全体最短 (赤)
    if all_dists:
        bx1, by1, bx2, by2, d_all = min(all_dists, key=lambda x: x[4])
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(frame, f"ALL D:{d_all:.2f}m", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # 真横最短 (緑)
    if side_dists:
        sx1, sy1, sx2, sy2, d_side = min(side_dists, key=lambda x: x[4])
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
        cv2.putText(frame, f"LAT D:{d_side:.2f}m", (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # RSS判定
    direction, rss = 'no detection', 0.0
    if len(roi_dists) >= 2:
        pts = np.array([[((x1 + x2) // 2), ((y1 + y2) // 2)] for x1, y1, x2, y2, _ in roi_dists])
        X = np.vstack([pts[:, 0], np.ones(len(pts))]).T
        Y = pts[:, 1]
        m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = np.abs(m * pts[:, 0] - pts[:, 1] + b) / math.sqrt(m * m + 1)
        rss = float(np.sum(resid**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'
    cv2.putText(frame, f"Dir:{direction} RSS:{int(rss)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 表示
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
cv2.destroyAllWindows()
