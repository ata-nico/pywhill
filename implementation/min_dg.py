#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO

# 実験用：最短距離ブロック描画＋録画

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
CAM_H             = 1.50
ERROR_OFFSET      = 0.0
LOOP_INTERVAL     = 0.1
OUTPUT_DIR        = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH       = OUTPUT_DIR + "\\experiment_minDg.mp4"
FPS               = 20.0

# --- 準備 ---
model = YOLO(YOLO_MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けません。")

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 'q' to quit.")

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    small = cv2.resize(frame, YOLO_INPUT_SIZE)
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # ROI 領域描画
    disp = frame.copy()
    x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * fw)
    x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * fw)
    cv2.rectangle(disp, (x1_roi, 0), (x2_roi, fh), (255, 0, 0), 2)

    # 全検出ブロックについて距離計算
    side_info = []  # [(x1,y1,x2,y2,Dg), ...]
    sw, sh = YOLO_INPUT_SIZE
    sx, sy = fw/sw, fh/sh
    for x1, y1, x2, y2 in boxes:
        # 真横距離計算
        y_max = max(y1, y2, (y1+y2)/2)
        phi = ((y_max/sh) - 0.5) * math.pi
        if abs(phi) < 1e-3:
            Dg = float('inf')
        else:
            Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
            Dg = max(Dg, 0.0)
        # スケール座標
        x1o, y1o = int(x1*sx), int(y1*sy)
        x2o, y2o = int(x2*sx), int(y2*sy)
        side_info.append((x1o, y1o, x2o, y2o, Dg))
        # 全検出矩形は薄いグリーンで描画
        cv2.rectangle(disp, (x1o, y1o), (x2o, y2o), (0, 255, 100), 1)

    # 最短距離ターゲット
    target = min(side_info, key=lambda t: t[4], default=None)
    if target:
        tx1, ty1, tx2, ty2, min_Dg = target
        # ターゲットブロックは赤で強調
        cv2.rectangle(disp, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        cv2.putText(disp,
                    f"Min Dg: {min_Dg:.2f}m",
                    (tx1, ty1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
    else:
        min_Dg = None

    # ステータス表示
    status = f"MinDg: {min_Dg if min_Dg is not None else 'N/A'}"
    cv2.putText(disp,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2)

    # 動画書き出し＆表示
    out.write(disp)
    cv2.imshow('Experiment MinDg', disp)

    # 終了判定
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ループ間隔
    elapsed = time.time() - start
    if elapsed < LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - elapsed)

# 後処理
cap.release()
out.release()
cv2.destroyAllWindows()
