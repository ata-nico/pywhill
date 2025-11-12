import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

# --- 設定 ---
VIDEO_PATH     = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video\test_movie_001.mp4"
MODEL_PATH     = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
ROI_ANGLE_DEG  = 60
FOV_DEG        = 360
CONF_THRESH    = 0.2
RSS_THRESH     = 2500

# --- 動画読み込み & モデルロード ---
cap = cv2.VideoCapture(VIDEO_PATH)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = YOLO(MODEL_PATH)

# --- 計測用変数 ---
frame_count     = 0
total_proc_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 計測スタート
    t0 = time.perf_counter()

    # 1) YOLO 推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(w, h))
    boxes   = results[0].boxes.xyxy.cpu().numpy()

    # 2) 重心抽出（ROI フィルタ込み）
    centers = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta_deg = (cx - w/2) / w * FOV_DEG
        if abs(theta_deg) <= ROI_ANGLE_DEG:
            cy = (y1 + y2) / 2
            centers.append((cx, cy))

    # 3) RSS 計算（最小二乗直線フィット＋RSS判定）
    if len(centers) >= 2:
        data = np.array(centers)
        X = np.vstack([data[:,0], np.ones(len(data))]).T
        Y = data[:,1]
        m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        numer     = np.abs(m*data[:,0] - data[:,1] + b)
        denom     = math.sqrt(m*m + 1)
        rss       = float(np.sum((numer/denom)**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'
    else:
        rss       = 0.0
        direction = 'no_detection'

    # 計測ストップ
    t1 = time.perf_counter()

    total_proc_time += (t1 - t0)
    frame_count     += 1

cap.release()

# 結果表示
avg_time_per_frame = total_proc_time / frame_count
fps = 1.0 / avg_time_per_frame

print(f"Processed frames      : {frame_count}")
print(f"Total processing time : {total_proc_time:.2f} s")
print(f"Average time/frame    : {avg_time_per_frame*1000:.2f} ms")
print(f"Estimated FPS         : {fps:.2f} (inference + centroid + RSS only)")
