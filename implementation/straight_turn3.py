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
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
CAM_H             = 1.55
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 50
TURN_SPEED_PCT    = 3
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.53

# --- 移動平均フィルタ設定 ---
WINDOW_SIZE = 5  # 平滑化ウィンドウ（フレーム数）
dg_history = deque(maxlen=WINDOW_SIZE)

# 出力動画設定
OUTPUT_DIR  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course0100_all_dist_draw_mavg.mp4"
FPS         = 60.0

# --- 状態履歴 ---
direction_history = deque(maxlen=20)
started = False
First_Dg = None  # 最初の距離を保持

# --- ヘルパー関数 ---
def drive_velocity(front_pct, side_pct=0):
    whill.send_velocity(int(front_pct/100*1000), int(side_pct/100*1500))

# --- カメラ＆モデル準備 ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けません。")
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

# VideoWriter 初期化
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 推論
    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # 画面サイズ＆ROI帯
    fh_, fw_ = frame.shape[:2]
    sx, sy = fw_/sw, fh_/sh
    x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * fw_)
    x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * fw_)

    # ROI内検出
    roi_info = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta = (cx - sw/2) / sw * FOV_DEG
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

    # 全ブロック距離計算
    side_info = []  # (x1,y1,x2,y2,Dg)
    for x1, y1, x2, y2 in boxes:
        y_max = max(y1, y2, (y1+y2)/2)
        phi = ((y_max/sh) - 0.5) * math.pi
        if abs(phi) < 1e-3:
            Dg = float('inf')
        else:
            Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
            Dg = max(Dg, 0.0)
        side_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy), Dg))

    # 最短距離ターゲット
    target = min(side_info, key=lambda t: t[4], default=None)
    if target:
        tx1, ty1, tx2, ty2, min_Dg = target
    else:
        min_Dg = None

    # 最初の距離を設定
    if target and First_Dg is None:
        First_Dg = min_Dg
        print(f"[INFO] First_Dg set to {First_Dg:.2f}m")

    # --- 移動平均フィルタ適用 ---
    if min_Dg is not None and not math.isinf(min_Dg):
        dg_history.append(min_Dg)
        avg_Dg = sum(dg_history) / len(dg_history)
    else:
        avg_Dg = None

    # ROI左右多数派
    roi_mid = (x1_roi + x2_roi) / 2
    left_count = sum(1 for (x1o,_,x2o,_) in roi_info if (x1o+x2o)/2 < roi_mid)
    right_count = len(roi_info) - left_count
    majority_side = 'Left' if left_count>right_count else 'Right' if right_count>left_count else 'Equal'

    # RANSAC inlier_ratio & direction
    direction, inlier_ratio, masks = 'no detection', 0.0, None
    if len(roi_info) >= 2:
        centres = np.array([[(x1+x2)/2,(y1+y2)/2] for x1,y1,x2,y2 in roi_info])
        X, Y = centres[:,0].reshape(-1,1), centres[:,1]
        ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
        ransac.fit(X, Y)
        masks = ransac.inlier_mask_
        inlier_ratio = masks.sum() / len(masks)
        direction = 'straight' if inlier_ratio>=INLIER_RATIO_TH else 'turn'
    direction_history.append(direction)

    # 開始前表示
    disp = frame.copy()
    if not started:
        cv2.putText(disp, "Press 's' to start", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('WHILL Control', disp)
        k = cv2.waitKey(1)&0xFF
        if k==ord('s'): started=True
        if k==ord('q'): break
        continue
    if cv2.waitKey(1)&0xFF==ord('q'): break

    # --- カスタム制御ロジック ---
    forward = 0
    side = 0
    if First_Dg is not None and avg_Dg is not None:
        lower_thr = First_Dg - 0.2
        upper_thr = First_Dg + 0.2

        # オフセット制御
        if majority_side == 'Right':
            if avg_Dg < lower_thr:
                for _ in range(int(0.5/LOOP_INTERVAL)):
                    drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT); time.sleep(LOOP_INTERVAL)
            elif avg_Dg > upper_thr:
                for _ in range(int(0.5/LOOP_INTERVAL)):
                    drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT); time.sleep(LOOP_INTERVAL)
            else:
                if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                    forward = FORWARD_SPEED_PCT
        elif majority_side == 'Left':
            if avg_Dg < lower_thr:
                for _ in range(int(0.5/LOOP_INTERVAL)):
                    drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT); time.sleep(LOOP_INTERVAL)
            elif avg_Dg > upper_thr:
                for _ in range(int(0.5/LOOP_INTERVAL)):
                    drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT); time.sleep(LOOP_INTERVAL)
            else:
                if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                    forward = FORWARD_SPEED_PCT
        else:
            if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                forward = FORWARD_SPEED_PCT

        # 表示更新
        disp_text = f"Ref:{First_Dg:.2f}(±0.20) avgDg:{avg_Dg:.2f}"
        cv2.putText(disp, disp_text, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    drive_velocity(forward, side)

    # --- 描画 ---
    cv2.rectangle(disp, (x1_roi,0), (x2_roi,fh_), (255,0,0), 2)
    if masks is not None:
        for i,(x1o,y1o,x2o,y2o) in enumerate(roi_info):
            color = (0,255,0) if masks[i] else (0,165,255)
            cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), color, 2)
    else:
        for x1o,y1o,x2o,y2o in roi_info:
            cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), (0,255,0), 2)

    if target:
        cv2.rectangle(disp, (tx1,ty1), (tx2,ty2), (0,0,255), 2)
        cv2.putText(disp, f"Dg:{avg_Dg if avg_Dg is not None else min_Dg:.2f}m", (tx1, ty1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    status = f"Dir:{direction} Inl:{inlier_ratio:.2f} Side:{majority_side}"
    cv2.putText(disp, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    out.write(disp)
    scale = min(MAX_WIN_W/fw_, MAX_WIN_H/fh_, 1.0)
    if scale<1.0:
        disp = cv2.resize(disp, (int(fw_*scale), int(fh_*scale)))
    cv2.imshow('WHILL Control', disp)

    dt = time.time() - t0
    if dt<LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - dt)

# --- 後処理 ---
drive_velocity(0,0)
whill.set_power(False)
cap.release()
out.release()
cv2.destroyAllWindows()
