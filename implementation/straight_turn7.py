#!/usr/bin/env python3
import os
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque
from datetime import datetime

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
# 起動待ち
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
FORWARD_SPEED_PCT = 30
TURN_SPEED_PCT    = 4
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.53

# --- 移動平均フィルタ設定 ---
WINDOW_SIZE = 5  # 平滑化ウィンドウ（フレーム数）
dg_history = deque(maxlen=WINDOW_SIZE)

# --- 評価ログ ---
times = []       # フレーム処理時間記録
errors = []      # 閾値帯からの距離ズレ記録

dt = LOOP_INTERVAL

# --- ログ保存設定 ---
LOG_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\kakukaku"
os.makedirs(LOG_DIR, exist_ok=True)
# タイムスタンプ取得
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- 出力動画設定 ---
OUTPUT_PATH = os.path.join(LOG_DIR, f"course_mavg_{timestamp}.mp4")

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
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 60.0, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

try:
    while True:
        start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推論
        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy()

        # ROI 計算
        fh_, fw_ = frame.shape[:2]
        sx, sy = fw_/sw, fh_/sh
        x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * fw_)
        x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * fw_)
        roi_info = []
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2
            theta = (cx - sw/2) / sw * FOV_DEG
            if abs(theta) <= ROI_ANGLE_DEG:
                roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

        # 全ブロック距離計算
        side_info = []
        for x1, y1, x2, y2 in boxes:
            y_max = max(y1, y2, (y1+y2)/2)
            phi = ((y_max/sh) - 0.5) * math.pi
            if abs(phi) < 1e-3:
                Dg = float('inf')
            else:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                Dg = max(Dg, 0.0)
            side_info.append(Dg)

        min_Dg = min(side_info) if side_info else None

        # 初期距離設定
        if min_Dg and First_Dg is None:
            First_Dg = min_Dg
            lower_thr = First_Dg - 0.2
            upper_thr = First_Dg + 0.2
            print(f"First_Dg={First_Dg:.2f}, Threshold=[{lower_thr:.2f}, {upper_thr:.2f}]")

        # 移動平均フィルタ
        if min_Dg is not None and not math.isinf(min_Dg):
            dg_history.append(min_Dg)
            avg_Dg = sum(dg_history) / len(dg_history)
        else:
            avg_Dg = None

        # 誤差ログ記録
        if avg_Dg is None:
            err = 0.0
        else:
            if avg_Dg < lower_thr:
                err = lower_thr - avg_Dg
            elif avg_Dg > upper_thr:
                err = avg_Dg - upper_thr
            else:
                err = 0.0
        errors.append(err)

        # ROI左右多数派
        roi_mid = (x1_roi + x2_roi) / 2
        left_count = sum(1 for (x1o,_,x2o,_) in roi_info if (x1o+x2o)/2 < roi_mid)
        right_count = len(roi_info) - left_count
        majority_side = 'Left' if left_count > right_count else 'Right' if right_count > left_count else 'Equal'

        # RANSACによる方向判定
        direction, inlier_ratio, masks = 'no detection', 0.0, None
        if len(roi_info) >= 2:
            pts = np.array([[(x1+x2)/2, (y1+y2)/2] for x1, y1, x2, y2 in roi_info])
            X, Y = pts[:,0].reshape(-1,1), pts[:,1]
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=RANSAC_THRESH_PX,
                random_state=0
            )
            ransac.fit(X, Y)
            masks = ransac.inlier_mask_
            inlier_ratio = masks.sum()/len(masks)
            direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'
        direction_history.append(direction)

        # 開始待ち
        disp = frame.copy()
        if not started:
            cv2.putText(disp, "Press 's' to start", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('WHILL Control', disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'): started = True
            if k == ord('q'): break
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # --- カスタム制御ロジック（変更なし）---
        forward = 0
        side = 0
        if First_Dg is not None and avg_Dg is not None:
            if majority_side == 'Right':
                if avg_Dg < lower_thr:
                    for _ in range(int(0.5/LOOP_INTERVAL)):
                        drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT)
                        time.sleep(LOOP_INTERVAL)
                elif avg_Dg > upper_thr:
                    for _ in range(int(0.5/LOOP_INTERVAL)):
                        drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT)
                        time.sleep(LOOP_INTERVAL)
                else:
                    if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                        forward = FORWARD_SPEED_PCT
            elif majority_side == 'Left':
                if avg_Dg < lower_thr:
                    for _ in range(int(0.5/LOOP_INTERVAL)):
                        drive_velocity(FORWARD_SPEED_PCT, TURN_SPEED_PCT)
                        time.sleep(LOOP_INTERVAL)
                elif avg_Dg > upper_thr:
                    for _ in range(int(0.5/LOOP_INTERVAL)):
                        drive_velocity(FORWARD_SPEED_PCT, -TURN_SPEED_PCT)
                        time.sleep(LOOP_INTERVAL)
                else:
                    if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                        forward = FORWARD_SPEED_PCT
            else:
                if direction_history.count('turn') < 10 and roi_info and inlier_ratio >= INLIER_RATIO_TH:
                    forward = FORWARD_SPEED_PCT

        drive_velocity(forward, side)

        # 描画・動画保存
        cv2.rectangle(disp, (x1_roi,0), (x2_roi,fh_), (255,0,0), 2)
        out.write(disp)
        cv2.imshow('WHILL Control', disp)

        # フレーム処理時間記録
        end = time.perf_counter()
        times.append(end - start)
        sleep = LOOP_INTERVAL - (end - start)
        if sleep > 0: time.sleep(sleep)

finally:
    # 後処理
    drive_velocity(0,0)
    whill.set_power(False)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ログ保存
    n = min(len(times), len(errors))
    if n > 0:
        summary_arr = np.column_stack((times[:n], errors[:n]))
        csv_path = os.path.join(LOG_DIR, f"log_mavg_{timestamp}.csv")
        np.savetxt(csv_path, summary_arr, delimiter=',', header='frame_time,error', comments='')
        avg_time = np.mean(times[:n])
        max_time = np.max(times[:n])
        std_time = np.std(times[:n])
        mse = np.mean(np.array(errors[:n])**2)
        ise = np.sum(np.array(errors[:n])**2) * dt
    else:
        csv_path = None
        avg_time = max_time = std_time = mse = ise = float('nan')

    summary_path = os.path.join(LOG_DIR, f"summary_mavg_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Avg frame time: {avg_time:.4f}s\n")
        f.write(f"Max frame time: {max_time:.4f}s\n")
        f.write(f"Std frame time: {std_time:.4f}s\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"ISE: {ise:.6f}\n")

    if csv_path:
        print(f"Logs saved to {csv_path} and {summary_path}")
    else:
        print(f"No data to save. Summary at {summary_path}")
