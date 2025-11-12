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
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
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

# --- 評価用ログ ---
times = []    # 各ループの実行時間記録
errors = []   # 閾値帯からのズレ

dt = LOOP_INTERVAL  # 制御周期

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

# VideoWriter（評価時は不要ならコメントアウト）
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output.mp4", fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))

print("Press 's' to start autonomous drive, 'q' to quit.")

try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推論
        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy()

        # ROI 内検出
        fh_, fw_ = frame.shape[:2]
        sx, sy = fw_/sw, fh_/sh
        x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * fw_)
        x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * fw_)
        roi_info = []
        for x1,y1,x2,y2 in boxes:
            cx = (x1+x2)/2
            theta = (cx - sw/2) / sw * FOV_DEG
            if abs(theta) <= ROI_ANGLE_DEG:
                roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

        # 距離計算
        side_info = []
        for x1,y1,x2,y2 in boxes:
            y_max = max(y1, y2, (y1+y2)/2)
            phi = ((y_max/sh) - 0.5) * math.pi
            if abs(phi) < 1e-3:
                Dg = float('inf')
            else:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                Dg = max(Dg, 0.0)
            side_info.append(Dg)

        min_Dg = min(side_info) if side_info else None

        # First_Dg 設定
        if min_Dg and First_Dg is None:
            First_Dg = min_Dg
            lower_thr = First_Dg - 0.2
            upper_thr = First_Dg + 0.2
            print(f"First_Dg = {First_Dg:.2f}, Threshold = [{lower_thr:.2f}, {upper_thr:.2f}]")

        # 移動平均
        if min_Dg is not None and not math.isinf(min_Dg):
            dg_history.append(min_Dg)
            avg_Dg = sum(dg_history) / len(dg_history)
        else:
            avg_Dg = None

        # 誤差
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

        # 制御（※挙動確認用。実際の制御ロジックを適宜置き換え）
        forward = 0
        if avg_Dg is not None:
            if avg_Dg < lower_thr:
                forward = FORWARD_SPEED_PCT
            elif avg_Dg > upper_thr:
                forward = FORWARD_SPEED_PCT
        drive_velocity(forward, 0)

        # 可視化／動画保存
        cv2.imshow('eval', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t_end = time.perf_counter()
        times.append(t_end - t_start)
        time.sleep(max(0, LOOP_INTERVAL - (t_end - t_start)))

finally:
    # 後処理
    drive_velocity(0,0)
    whill.set_power(False)
    cap.release()
    cv2.destroyAllWindows()

    # 評価結果出力
    times = np.array(times)
    errors = np.array(errors)
    mse = np.mean(errors**2)
    ise = np.sum(errors**2) * dt
    print(f"Avg frame time: {times.mean():.4f}s, Max: {times.max():.4f}s, Std: {times.std():.4f}s")
    print(f"MSE: {mse:.6f}, ISE: {ise:.6f}")
