import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- 設定 ---
VIDEO_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\mp4_weelchair\panorama_003.mp4"  # 入力動画パス
MODEL_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans17\weights\best.pt"                    # YOLOモデルパス
ROI_ANGLE_DEG     = 60                           # 中心部±ROI角度 [deg]
FOV_DEG           = 360                          # パノラマの水平視野角
CONF_THRESH       = 0.2                          # YOLO検出信頼度閾値
RANSAC_THRESH_PX  = 5.0                          # RANSAC residual_threshold (px)
INLIER_RATIO_TH   = 0.7                          # インライア比率閾値
OUTPUT_DIR       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"    # 保存先ディレクトリ
OUTPUT_VIDEO      = os.path.join(OUTPUT_DIR, "annotated_output_ransac.mp4")

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 動画読み込み & ライター設定 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# モデルロード
model = YOLO(MODEL_PATH)

# ROI領域計算
roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)

# --- フレーム処理ループ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(w, h))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # ROI内で重心抽出
    centers = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta_deg = (cx - w/2) / w * FOV_DEG
        if abs(theta_deg) <= ROI_ANGLE_DEG:
            cy = (y1 + y2) / 2
            centers.append((cx, cy, int(x1), int(y1), int(x2), int(y2)))

    # RANSAC フィット
    direction = 'no detection'
    inlier_ratio = 0.0
    if len(centers) >= 2:
        data = np.array([[c[0], c[1]] for c in centers])
        X = data[:,0].reshape(-1, 1)
        Y = data[:,1]
        ransac = RANSACRegressor(LinearRegression(),
                                residual_threshold=RANSAC_THRESH_PX,
                                random_state=0)
        ransac.fit(X, Y)
        mask = ransac.inlier_mask_
        inlier_ratio = mask.sum() / len(data)
        direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

    # 描画
    for cx, cy, x1, y1, x2, y2 in centers:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.rectangle(frame, (roi_x1, 0), (roi_x2, h), (255,0,0), 2)
    text = f"Dir: {direction} (Inlier: {inlier_ratio:.2f})"
    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # 書き込み
    out.write(frame)

# リソース解放
cap.release()
out.release()
print(f"Annotated video saved to: {OUTPUT_VIDEO}")
