import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
VIDEO_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\mp4_weelchair\panorama_003.mp4"
MODEL_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
ROI_ANGLE_DEG    = 60     # 中心部±ROI角度 [deg]
FOV_DEG          = 360    # パノラマ水平視野角 [deg]
CONF_THRESH      = 0.2    # YOLO検出信頼度
RSS_THRESH       = 10000  # RSS判定閾値
SIDE_THRESH_DEG  = 10     # 真横判定許容 (90±10°)
CAM_H            = 1.55   # カメラ地上高 [m]
ERROR_OFFSET     = 0.12   # 地面距離補正値 [m]
OUTPUT_DIR       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_VIDEO     = os.path.join(OUTPUT_DIR, "annotated_output_1m_01.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 動画＆ライター設定 ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# --- モデルロード ---
model = YOLO(MODEL_PATH)

# ROI範囲（中央±ROI_ANGLE_DEG）
roi_x1 = int((w/2) - (ROI_ANGLE_DEG/ FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG/ FOV_DEG) * w)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    results = model.predict(frame, conf=CONF_THRESH, imgsz=(w, h))
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # ROI内のセンター抽出 & 真横ブロックの地面距離推定
    centers   = []
    side_info = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        theta_deg = (cx - w/2) / w * FOV_DEG

        # ROI内なら走行方向推定用
        if abs(theta_deg) <= ROI_ANGLE_DEG:
            centers.append((int(x1), int(y1), int(x2), int(y2)))

        # 真横(約90°)を検出して地面距離を計算
        if abs(abs(theta_deg) - 90.0) <= SIDE_THRESH_DEG:
            # ブロックの底面Y: [y1, y2, cy]の最大値
            y_max = max(y1, y2, cy)
            # 全天球パノラマの鉛直角に変換
            phi_rad = ((y_max / h) - 0.5) * math.pi
            if abs(phi_rad) < 1e-3:
                D_ground = float('inf')
            else:
                D_ground = CAM_H / math.tan(abs(phi_rad))
                D_ground = max(D_ground - ERROR_OFFSET, 0.0)
            side_info.append((int(x1), int(y1), int(x2), int(y2), D_ground))

    # RSSによる直進/旋回判定
    direction = 'no detection'
    rss = 0.0
    if len(centers) >= 2:
        pts = np.array([[ (x1+x2)/2, (y1+y2)/2 ] for x1,y1,x2,y2 in centers])
        A   = np.vstack([pts[:,0], np.ones(len(pts))]).T
        m,b = np.linalg.lstsq(A, pts[:,1], rcond=None)[0]
        resid = np.abs(m*pts[:,0] - pts[:,1] + b) / math.sqrt(m*m+1)
        rss   = float(np.sum(resid**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'

    # 描画
    # ROI枠
    cv2.rectangle(frame, (roi_x1,0), (roi_x2,h), (255,0,0), 2)
    # ROI内センター
    for x1,y1,x2,y2 in centers:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # 真横ブロック & 地面距離
    for x1,y1,x2,y2,Dg in side_info:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 3)
        cv2.putText(frame, f"G:{Dg:.2f}m", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    # 方向テキスト
    cv2.putText(frame, f"Dir: {direction} (RSS={int(rss)})",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved to: {OUTPUT_VIDEO}")
