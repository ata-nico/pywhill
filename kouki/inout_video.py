#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression
from datetime import datetime
import os

# --- 設定 ---
# ★★★ ここに検証したい動画のフルパスを記述してください ★★★
# 例: r"C:\Users\YourUser\Desktop\test_video.mp4"
# カメラを使う場合は None のままにしてください
INPUT_VIDEO_PATH = r"C:\Users\ata3357\Desktop\zemi_win\mp4_weelchair\panorama_001.mp4"

CAMERA_ID       = 0
ROI_ANGLE_DEG   = 60
FOV_DEG         = 360
CONF_THRESH     = 0.2
SIDE_THRESH_DEG = 10
CAM_H           = 1.55
YOLO_INPUT_SIZE = (640, 480)
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
RANSAC_THRESH_PX= 5.0
INLIER_RATIO_TH = 0.53
FRONT_TH_DEG    = 15.0 # 「正面」と見なす角度のしきい値

OUTPUT_DIR      = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\video"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path      = os.path.join(OUTPUT_DIR, f"processed_{timestamp}.mp4")

# --- ウィンドウ ---
WINDOW_NAME = 'Course Type Detection'

# --- モデルロード ---
model = YOLO(MODEL_PATH)

# --- 入力ソースの選択 (動画ファイル or カメラ) ---
if INPUT_VIDEO_PATH:
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    print(f"[INFO] Processing video file: {INPUT_VIDEO_PATH}")
else:
    cap = cv2.VideoCapture(CAMERA_ID)
    print(f"[INFO] Using camera ID: {CAMERA_ID}")

if not cap.isOpened():
    raise RuntimeError(f"Input source could not be opened.")

# --- 動画保存用設定 ---
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30.0 # 動画ファイルの場合、FPSが0になることがあるためデフォルト値を設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (fw, fh))

print(f"[INFO] Processed video will be saved to: {video_path}")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video file.")
        break

    h, w = frame.shape[:2]
    sw, sh = YOLO_INPUT_SIZE
    scale_x = w / sw
    scale_y = h / sh

    # 正面判定の閾値ラインを描画 (マゼンタ)
    x_left_thresh = int((0.5 - FRONT_TH_DEG / FOV_DEG) * w)
    x_right_thresh = int((0.5 + FRONT_TH_DEG / FOV_DEG) * w)
    cv2.line(frame, (x_left_thresh, 0), (x_left_thresh, h), (255, 0, 255), 1)
    cv2.line(frame, (x_right_thresh, 0), (x_right_thresh, h), (255, 0, 255), 1)

    # ★★★ 追加: ROIの範囲を描画 ★★★
    x_roi_left = int((0.5 - ROI_ANGLE_DEG / FOV_DEG) * w)
    x_roi_right = int((0.5 + ROI_ANGLE_DEG / FOV_DEG) * w)
    # ROIラインを描画 (シアン色)
    cv2.line(frame, (x_roi_left, 0), (x_roi_left, h), (255, 255, 0), 1)
    cv2.line(frame, (x_roi_right, 0), (x_roi_right, h), (255, 255, 0), 1)
    # ★★★ここまで★★★

    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    all_dists = []
    roi_info = []
    side_dists = []

    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        theta = ((cx / sw) - 0.5) * FOV_DEG
        y_max = max(cy, y1, y2)
        phi = ((y_max / sh) - 0.5) * math.pi
        Dg = float('inf') if abs(phi) < 1e-3 else CAM_H / math.tan(abs(phi))

        rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
        rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
        all_dists.append((rx1, ry1, rx2, ry2, Dg))

        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((theta, Dg, rx1, ry1, rx2, ry2))
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            side_dists.append((rx1, ry1, rx2, ry2, Dg))

    # ROI枠内のブロックを強調 (白)
    for theta, Dg, x1, y1, x2, y2 in roi_info:
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

    # RANSAC方向推定
    direction = 'no detection'
    inlier_ratio = 0.0
    if len(roi_info) >= 2:
        pts = np.array([[ (x1 + x2) / 2, (y1 + y2) / 2 ] for _, _, x1, y1, x2, y2 in roi_info])
        X = pts[:, 0].reshape(-1, 1)
        Y = pts[:, 1]
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
        ransac.fit(X, Y)
        inliers = ransac.inlier_mask_
        inlier_ratio = inliers.sum() / len(inliers)
        direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

    cv2.putText(frame, f"Dir:{direction} InlierRatio:{inlier_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # インコース／アウトコース判定
    if direction == 'turn':
        course_type = "OUT-COURSE" # デフォルトはアウトコース
        front_block = None
        
        if roi_info:
            candidate_front = min(roi_info, key=lambda x: abs(x[0]))
            
            if abs(candidate_front[0]) <= FRONT_TH_DEG:
                front_block = candidate_front
                course_type = "IN-COURSE"

        color = (0, 255, 255) if course_type == "IN-COURSE" else (255, 0, 255)
        cv2.putText(frame, course_type, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if course_type == "IN-COURSE" and front_block is not None:
            _, d_front, fx1, fy1, fx2, fy2 = front_block
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            cv2.putText(frame, f"FRONT D:{d_front:.2f}m", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 表示・保存
    out.write(frame)
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Processed video saved.")
