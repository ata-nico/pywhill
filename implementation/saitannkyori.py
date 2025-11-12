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
RSS_THRESH     = 2500
CAM_H          = 1.55
OUTPUT_DIR     = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
VIDEO_OUT      = os.path.join(OUTPUT_DIR, "saitankyori_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4")
FPS            = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
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

    # ROI内と全検出で距離リスト作成
    roi_dists = []      # ROI内の (x1,y1,x2,y2,Dg)
    all_dists = []      # 全検出の (x1,y1,x2,y2,Dg)

    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        theta = ((cx - width/2) / width) * FOV_DEG
        # 距離計算
        y_max = max(y1, y2, cy)
        phi = ((y_max / height) - 0.5) * math.pi
        Dg = float('inf') if abs(phi) < 1e-3 else CAM_H / math.tan(abs(phi))
        # 全検出に追加
        all_dists.append((x1, y1, x2, y2, Dg))
        # ROI内判定
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_dists.append((x1, y1, x2, y2, Dg))

    # ROI内すべて表示 (白枠)
    for x1, y1, x2, y2, _ in roi_dists:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)

    # 全検出中最短距離を赤枠表示
    if all_dists:
        nearest_all = min(all_dists, key=lambda x: x[4])
        x1, y1, x2, y2, Dg = nearest_all
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(frame, f"ALL D:{Dg:.2f}m", (int(x1), int(y1)-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # RSS判定
    centers = [(x1, y1, x2, y2) for x1,y1,x2,y2,_ in roi_dists]
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
    cv2.putText(frame, f"Dir:{direction} RSS:{int(rss)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # フレーム出力
    out.write(frame)
    cv2.imshow("Realtime Panorama", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved video: {VIDEO_OUT}")
