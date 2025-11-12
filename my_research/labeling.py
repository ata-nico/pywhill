import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import time
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- 設定 ---
VIDEO_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video\test_movie_001.mp4"
MODEL_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans19\weights\best.pt"
ROI_ANGLE_DEG    = 60
FOV_DEG          = 360
CONF_THRESH      = 0.2
RANSAC_THRESH_PX = 5.0
INLIER_RATIO_TH  = 0.55

# --- フレーム保存用ディレクトリ ---
SAVE_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\label"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 動画読み込み & モデルロード ---
cap = cv2.VideoCapture(VIDEO_PATH)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = YOLO(MODEL_PATH)

roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)

# --- 計測開始変数 ---
start_time  = time.time()
frame_count = 0
frame_idx   = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論・重心抽出
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(w, h))
    boxes   = results[0].boxes.xyxy.cpu().numpy()
    centers = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta_deg = (cx - w/2) / w * FOV_DEG
        if abs(theta_deg) <= ROI_ANGLE_DEG:
            cy = (y1 + y2) / 2
            centers.append((cx, cy, int(x1), int(y1), int(x2), int(y2)))

    # RANSAC フィット
    direction    = 'no_detection'
    inlier_ratio = 0.0
    if len(centers) >= 2:
        data = np.array([[c[0], c[1]] for c in centers])
        X = data[:,0].reshape(-1, 1)
        Y = data[:,1]
        ransac = RANSACRegressor(
            LinearRegression(),
            residual_threshold=RANSAC_THRESH_PX,
            random_state=0
        )
        ransac.fit(X, Y)
        mask = ransac.inlier_mask_
        inlier_ratio = mask.sum() / len(data)
        direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

    # **2フレームごとに保存処理**
    if frame_idx % 3 == 0:
        # --- 生画像の保存 ---
        raw_fname = os.path.join(
            SAVE_DIR,
            f"frame_{frame_idx:06d}_{direction}_raw.png"
        )
        cv2.imwrite(raw_fname, frame)

        # --- アノテーション画像の作成 & 保存 ---
        ann_frame = frame.copy()
        # バウンディングボックス、ROI、テキスト描画
        for cx, cy, x1, y1, x2, y2 in centers:
            cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.rectangle(ann_frame, (roi_x1, 0), (roi_x2, h), (255,0,0), 2)
        text = f"Dir:{direction} Inlier:{inlier_ratio:.2f}"
        cv2.putText(ann_frame, text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        ann_fname = os.path.join(
            SAVE_DIR,
            f"frame_{frame_idx:06d}_{direction}_annotated.png"
        )
        cv2.imwrite(ann_fname, ann_frame)

    frame_count += 1
    frame_idx   += 1

# --- 後処理 & 計測結果表示 ---
cap.release()
elapsed = time.time() - start_time
avg_fps = frame_count / elapsed

print(f"Processed frames : {frame_count}")
print(f"Elapsed time     : {elapsed:.2f} s")
print(f"Average FPS      : {avg_fps:.2f}")
print(f"Saved to         : {SAVE_DIR}")
