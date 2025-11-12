import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
IMAGE_DIR       = r"C:\Users\ata3357\Desktop\zemi_win\panorama\aaa"  # 検証用画像ディレクトリ
OUTPUT_DIR      = r"C:\Users\ata3357\Desktop\zemi_win\test_output"       # 結果保存先ディレクトリ
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
ROI_ANGLE_DEG   = 60    # 前方視野中心±ROI角度 [deg]
FOV_DEG         = 360   # パノラマ水平視野角 [deg]
CONF_THRESH     = 0.2   # YOLO信頼度閾値
RSS_THRESH      = 2500  # RSS閾値
SIDE_THRESH_DEG = 10    # 真横検出閾値 (|θ - 90| <= 閾値)
CAM_H           = 1.55  # カメラ高さ [m]

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデルロード
model = YOLO(MODEL_PATH)

# テスト画像ループ
for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(IMAGE_DIR, fname)
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    height, width = frame.shape[:2]

    # YOLO推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(width, height))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 距離リスト作成
    all_dists, roi_dists, side_info = [], [], []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        theta = ((cx - width/2) / width) * FOV_DEG
        y_max = max(y1, y2, cy)
        phi = ((y_max/height) - 0.5) * math.pi
        Dg = float('inf') if abs(phi) < 1e-3 else CAM_H / math.tan(abs(phi))
        all_dists.append((x1, y1, x2, y2, Dg))
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_dists.append((x1, y1, x2, y2, Dg))
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            side_info.append((x1, y1, x2, y2, Dg))

    # アノテーション描画
    # 前方ROIを白枠、全検出最短距離を赤枠、真横最短距離を緑枠
    for x1, y1, x2, y2, _ in roi_dists:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)
    if all_dists:
        x1, y1, x2, y2, Dg = min(all_dists, key=lambda x: x[4])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(frame, f"ALL D:{Dg:.2f}m", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    if side_info:
        x1, y1, x2, y2, Dg = min(side_info, key=lambda x: x[4])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"LAT D:{Dg:.2f}m", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # RSS 判定
    direction, rss = 'no detection', 0.0
    if len(roi_dists) >= 2:
        pts = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2,_ in roi_dists])
        X = np.vstack([pts[:,0], np.ones(len(pts))]).T
        Y = pts[:,1]
        m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = np.abs(m*pts[:,0] - pts[:,1] + b) / math.sqrt(m*m + 1)
        rss = float(np.sum(resid**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'
    cv2.putText(frame, f"Dir:{direction} RSS:{int(rss)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # 結果保存
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, frame)
    print(f"Saved: {out_path}")
