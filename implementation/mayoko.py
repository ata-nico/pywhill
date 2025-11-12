import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
CAMERA_ID      = 0  # カメラID
MODEL_PATH     = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
ROI_ANGLE_DEG  = 60  # 中心部±ROI角度 [deg]
FOV_DEG        = 360 # パノラマ水平視野角 [deg]
CONF_THRESH    = 0.2 # YOLO閾値
RSS_THRESH     = 10000
REAL_WIDTH     = 0.2
SIDE_THRESH_DEG= 10
CAM_H          = 1.55
OUTPUT_DIR     = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
VIDEO_OUT      = os.path.join(OUTPUT_DIR, "mayoko_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")
FPS            = 10

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデルロード
model = YOLO(MODEL_PATH)

# カメラ & VideoWriter設定
cap = cv2.VideoCapture(CAMERA_ID)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(width, height))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    centers = []                   # ROI内ボックス
    side_info = []                 # 真横ボックスと距離

    # 検出 & 距離推定
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        theta_h = ((cx - width/2) / width) * FOV_DEG

        # ROI内
        if abs(theta_h) <= ROI_ANGLE_DEG:
            centers.append((x1, y1, x2, y2))

        # 真横検出
        if abs(abs(theta_h) - 90.0) <= SIDE_THRESH_DEG:
            y_max = max(y1, y2, cy)
            phi_rad = ((y_max / height) - 0.5) * math.pi
            D_ground = float('inf') if abs(phi_rad) < 1e-3 else CAM_H / math.tan(abs(phi_rad))
            side_info.append((x1, y1, x2, y2, D_ground))

    # ROI内すべて表示 (白枠)
    for x1, y1, x2, y2 in centers:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)

    # 真横のうち最短距離の点字ブロックだけ強調 (緑枠+距離)
    if side_info:
        nearest = min(side_info, key=lambda x: x[4])
        x1, y1, x2, y2, Dg = nearest
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(frame, f"G:{Dg:.2f}m", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # RSS判定
    direction = 'no detection'
    rss = 0.0
    if len(centers) >= 2:
        pts = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2 in centers])
        X = np.vstack([pts[:,0], np.ones(len(pts))]).T
        Y = pts[:,1]
        m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = np.abs(m*pts[:,0] - pts[:,1] + b) / math.sqrt(m*m + 1)
        rss = float(np.sum(resid**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'

    # ROI枠表示
    rx1 = int((width/2) - (ROI_ANGLE_DEG/FOV_DEG)*width)
    rx2 = int((width/2) + (ROI_ANGLE_DEG/FOV_DEG)*width)
    cv2.rectangle(frame, (rx1,0), (rx2,height), (255,0,0), 1)

    # 情報表示
    cv2.putText(frame, f"Dir:{direction} RSS:{int(rss)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # フレーム出力
    out.write(frame)
    cv2.imshow("Realtime Panorama", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# クリーンアップ
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved video: {VIDEO_OUT}")
