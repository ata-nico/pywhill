import cv2
import numpy as np
from ultralytics import YOLO
import time
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- 設定 ---
VIDEO_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video\test_movie_001.mp4"
MODEL_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
ROI_ANGLE_DEG    = 60
FOV_DEG          = 360
CONF_THRESH      = 0.2
RANSAC_THRESH_PX = 5.0
INLIER_RATIO_TH  = 0.53

# --- 動画読み込み & モデルロード ---
cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = YOLO(MODEL_PATH)

# ROI の左右端
roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)

# 計測用変数
frame_count     = 0
total_proc_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # タイミング計測スタート
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

    # 3) RANSAC フィット
    if len(centers) >= 2:
        data = np.array(centers)
        X = data[:,0].reshape(-1,1)
        Y = data[:,1]
        ransac = RANSACRegressor(
            LinearRegression(),
            residual_threshold=RANSAC_THRESH_PX,
            random_state=0
        )
        ransac.fit(X, Y)
        # 必要なら inlier_ratio = ransac.inlier_mask_.sum() / len(data)

    # タイミング計測ストップ
    t1 = time.perf_counter()

    proc_time = t1 - t0
    total_proc_time += proc_time
    frame_count     += 1

# 後処理
cap.release()

# 結果表示
avg_time_per_frame = total_proc_time / frame_count
fps = 1.0 / avg_time_per_frame

print(f"Processed frames        : {frame_count}")
print(f"Total processing time   : {total_proc_time:.2f} sec")
print(f"Average time per frame  : {avg_time_per_frame*1000:.2f} ms")
print(f"Estimated FPS (inference+centroid+RANSAC only): {fps:.2f}")
