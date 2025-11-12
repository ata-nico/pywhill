import os
import csv
import math
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- 設定 ---
IMG_DIR         = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\raw_only"            # raw画像があるフォルダ
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
CONF_THRESH     = 0.2
ROI_ANGLE_DEG   = 60
FOV_DEG         = 360
RANSAC_THRESH_PX= 5.0

# YOLOモデルロード
model = YOLO(MODEL_PATH)

def calc_scores(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 推論＋重心抽出
    res    = model.predict(img, conf=CONF_THRESH, imgsz=(w,h))[0]
    boxes  = res.boxes.xyxy.cpu().numpy()
    centers = []
    for x1,y1,x2,y2 in boxes:
        cx = (x1+x2)/2
        theta = (cx - w/2)/w * FOV_DEG
        if abs(theta) <= ROI_ANGLE_DEG:
            cy = (y1+y2)/2
            centers.append((cx, cy))

    # RANSAC スコア
    if len(centers) >= 2:
        data = np.array(centers)
        X = data[:,0].reshape(-1,1)
        Y = data[:,1]
        ransac = RANSACRegressor(
            LinearRegression(),
            residual_threshold=RANSAC_THRESH_PX,
            random_state=0
        ).fit(X, Y)
        inlier_ratio = ransac.inlier_mask_.sum() / len(data)
    else:
        inlier_ratio = 0.0

    # RSS スコア
    if len(centers) >= 2:
        data = np.array(centers)
        A = np.vstack([data[:,0], np.ones(len(data))]).T
        B = data[:,1]
        m, b = np.linalg.lstsq(A, B, rcond=None)[0]
        dist = np.abs(m*data[:,0] - data[:,1] + b) / math.sqrt(m*m + 1)
        rss = float((dist**2).sum())
    else:
        rss = float('inf')

    return inlier_ratio, rss

# --- CSV に書き出し ---
csv_path = os.path.join(IMG_DIR, "score_results.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename","gt_label","inlier_ratio","rss"])
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.endswith("_raw.png"):
            continue
        gt_label = fname.split("_")[2]
        ir_score, rss_score = calc_scores(os.path.join(IMG_DIR, fname))
        writer.writerow([fname, gt_label, f"{ir_score:.4f}", f"{rss_score:.1f}"])

print(f"Scores written to {csv_path}")
