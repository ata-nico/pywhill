import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
import time

# --- 設定 ---
VIDEO_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\mp4_weelchair\panorama_003.mp4"
MODEL_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans7\weights\best.pt"
ROI_ANGLE_DEG    = 60
FOV_DEG          = 360
CONF_THRESH      = 0.2
RSS_THRESH       = 1500

OUTPUT_DIR       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_VIDEO     = os.path.join(OUTPUT_DIR, "annotated_output_v2.mp4")
FRAME_SAVE_DIR   = os.path.join(OUTPUT_DIR, "frames_for_labeling")

# --- 切り替えフラグ ---
SAVE_FRAMES = False   # ← True にするとフレームPNG保存も行う

# --- 準備 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_FRAMES:
    os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (w, h))

model = YOLO(MODEL_PATH)
roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)

# --- FPS計測開始 ---
start_time = time.time()
frame_count = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(w, h))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 重心抽出
    centers = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta_deg = (cx - w/2) / w * FOV_DEG
        if abs(theta_deg) <= ROI_ANGLE_DEG:
            cy = (y1 + y2) / 2
            centers.append((cx, cy, int(x1), int(y1), int(x2), int(y2)))

    # RSS 判定
    direction = 'no_detection'
    rss = 0.0
    if len(centers) >= 2:
        data = np.array([[c[0], c[1]] for c in centers])
        X = np.vstack([data[:,0], np.ones(len(data))]).T
        Y = data[:,1]
        m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        numer = np.abs(m*data[:,0] - data[:,1] + b)
        denom = math.sqrt(m*m + 1)
        distances = numer / denom
        rss = float(np.sum(distances**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'

    # 描画
    for cx, cy, x1, y1, x2, y2 in centers:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.rectangle(frame, (roi_x1, 0), (roi_x2, h), (255,0,0), 2)
    text = f"Dir:{direction} RSS={int(rss)}"
    cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # --- 出力切り替え ---
    out.write(frame)
    if SAVE_FRAMES:
        save_name = f"frame_{frame_idx:06d}_{direction}.png"
        cv2.imwrite(os.path.join(FRAME_SAVE_DIR, save_name), frame)

    frame_count += 1
    frame_idx  += 1

# --- 後処理 & FPS計測終了 ---
cap.release()
out.release()
elapsed = time.time() - start_time
avg_fps = frame_count / elapsed

print(f"Processed frames : {frame_count}")
print(f"Elapsed time     : {elapsed:.2f} s")
print(f"Average FPS      : {avg_fps:.2f}")
print(f"Annotated video  : {OUTPUT_VIDEO}")
if SAVE_FRAMES:
    print(f"Frames saved to  : {FRAME_SAVE_DIR}")
