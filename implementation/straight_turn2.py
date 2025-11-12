#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
SIDE_THRESH_DEG   = 10
CAM_H             = 1.50
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 20
TURN_SPEED_PCT    = 5
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.55

# 出力動画設定
OUTPUT_DIR  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course013.mp4"
FPS         = 20.0

# --- 状態履歴 ---
direction_history = deque(maxlen=20)

started = False

def drive_velocity(front_pct, side_pct=0):
    whill.send_velocity(int(front_pct/100*1000),
                        int(side_pct/100*1500))

# カメラ＆モデル準備
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けません。")
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

# VideoWriter 初期化
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    small = cv2.resize(frame, (sw, sh))
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # 画面サイズ＆ROI帯
    fh_, fw_ = frame.shape[:2]
    sx, sy   = fw_/sw, fh_/sh
    x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * sw * sx)
    x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * sw * sx)

    # ROI内検出 & 真横検出距離リスト
    roi_info  = []
    side_info = []  # ここで5要素タプルを必ず append
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta = (cx - sw/2) / sw * FOV_DEG

        # 前方 ROI
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

        # 真横判定＋距離計算
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            y_max = max(y1, y2, (y1+y2)/2)
            phi = ((y_max/sh) - 0.5) * math.pi
            if abs(phi) < 1e-3:
                Dg = float('inf')
            else:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                Dg = max(Dg, 0.0)
            # ここで5要素タプルを追加
            side_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), Dg))

    # 真横距離の最小値
    min_Dg = min((d for *_, d in side_info), default=None)

    # ROI左右多数派判定
    roi_mid = (x1_roi + x2_roi) / 2
    left_count  = sum(1 for (x1o,_,x2o,_) in roi_info if (x1o+x2o)/2 < roi_mid)
    right_count = len(roi_info) - left_count
    if   left_count  > right_count:
        majority_side = 'Left'
    elif right_count > left_count:
        majority_side = 'Right'
    else:
        majority_side = 'Equal'

    # RANSAC inlier_ratio & direction
    direction    = 'no detection'
    inlier_ratio = 0.0
    masks        = None
    if len(roi_info) >= 2:
        centers = np.array([[(x1+x2)/2,(y1+y2)/2] for x1,y1,x2,y2 in roi_info])
        X = centers[:,0].reshape(-1,1)
        Y = centers[:,1]
        ransac = RANSACRegressor(LinearRegression(),
                                 residual_threshold=RANSAC_THRESH_PX,
                                 random_state=0)
        ransac.fit(X, Y)
        masks        = ransac.inlier_mask_
        inlier_ratio = masks.sum() / len(masks)
        direction    = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

    direction_history.append(direction)

    # 開始前表示
    disp = frame.copy()
    if not started:
        cv2.putText(disp, "Press 's' to start", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('WHILL Control', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'): started = True
        if k == ord('q'): break
        continue

    # --- カスタム制御ロジック ---
    forward = 0

    if majority_side == 'Right' and min_Dg is not None:
        if min_Dg < 0.80:
            # 時計回り 0.5 秒
            for _ in range(int(0.5/LOOP_INTERVAL)):
                drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT)
                time.sleep(LOOP_INTERVAL)
        elif min_Dg >= 1.20:
            # 反時計回り 0.5 秒
            for _ in range(int(0.5/LOOP_INTERVAL)):
                drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT)
                time.sleep(LOOP_INTERVAL)
        else:
            if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                forward = FORWARD_SPEED_PCT

    elif majority_side == 'Left' and min_Dg is not None:
        if min_Dg < 0.80:
            # 反時計回り 0.5 秒
            for _ in range(int(0.5/LOOP_INTERVAL)):
                drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT)
                time.sleep(LOOP_INTERVAL)
        elif min_Dg >= 1.20:
            # 時計回り 0.5 秒
            for _ in range(int(0.5/LOOP_INTERVAL)):
                drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT)
                time.sleep(LOOP_INTERVAL)
        else:
            if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                forward = FORWARD_SPEED_PCT

    else:
        if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
            forward = FORWARD_SPEED_PCT

    # 最終前進コマンド
    drive_velocity(forward, 0)

    # --- 描画 ---
    cv2.rectangle(disp, (x1_roi,0), (x2_roi,fh_), (255,0,0), 2)
    if masks is not None:
        for i,(x1o,y1o,x2o,y2o) in enumerate(roi_info):
            c = (0,255,0) if masks[i] else (0,165,255)
            cv2.rectangle(disp,(x1o,y1o),(x2o,y2o),c,2)
    else:
        for x1o,y1o,x2o,y2o in roi_info:
            cv2.rectangle(disp,(x1o,y1o),(x2o,y2o),(0,255,0),2)

    for x1o,y1o,x2o,y2o,Dg in side_info:
        cv2.rectangle(disp,(x1o,y1o),(x2o,y2o),(0,255,255),2)
        cv2.putText(disp,f"{Dg:.2f}m",(x1o,y1o-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    status = f"{direction} Inl:{inlier_ratio:.2f} Side:{majority_side}"
    cv2.putText(disp,status,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)

    out.write(disp)
    scale = min(MAX_WIN_W/fw_, MAX_WIN_H/fh_, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp,(int(fw_*scale),int(fh_*scale)),
                          interpolation=cv2.INTER_AREA)
    cv2.imshow('WHILL Control', disp)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

    dt = time.time() - t0
    if dt < LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - dt)

# --- 後処理 ---
drive_velocity(0,0)
whill.set_power(False)
cap.release()
out.release()
cv2.destroyAllWindows()
