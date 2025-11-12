#!/usr/bin/env python3

import time
import math
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from whill import ComWHILL

# --- WHILL 初期化 ---
COM_PORT = 'COM3'
whill = ComWHILL(port=COM_PORT)
whill.send_power_on()
whill.set_battery_voltage_output_mode(vbatt_on_off=True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID        = 0
YOLO_MODEL_PATH  = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE  = (640, 480)
CONF_THRESH      = 0.2

ROI_ANGLE_DEG    = 60
FOV_DEG          = 360
RSS_THRESH       = 10000

CAM_H            = 1.55
ERROR_OFFSET     = 0.12
TARGET_DISTANCE  = 1.0
DIST_THRESH      = 0.25

SPEED_VALUE      = 20
ANGLE_CORR_DUR   = 1.0
CORR_INTERVAL    = 0.1

HISTORY_LEN      = 5
history = deque(maxlen=HISTORY_LEN)

# ウィンドウ最大表示サイズ
MAX_WIN_W, MAX_WIN_H = 800, 600

# --- WHILL 制御ヘルパー ---
def drive_velocity(front_pct: int, side_pct: int):
    sdk_front = int((front_pct / 100) * 1000)
    sdk_side  = int((side_pct  / 100) * 1500)
    whill.send_velocity(sdk_front, sdk_side)

def correct_angle_loop(distance: float, side_flag: str) -> bool:
    if abs(distance - TARGET_DISTANCE) <= DIST_THRESH:
        return False
    if side_flag == 'left':
        cmd_side = -SPEED_VALUE if distance > TARGET_DISTANCE else SPEED_VALUE
    else:
        cmd_side =  SPEED_VALUE if distance > TARGET_DISTANCE else -SPEED_VALUE
    count = int(ANGLE_CORR_DUR / CORR_INTERVAL)
    for _ in range(count):
        drive_velocity(0, cmd_side)
        time.sleep(CORR_INTERVAL)
    drive_velocity(0, 0)
    return True

# --- カメラ & モデル準備 ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けませんでした。")
model = YOLO(YOLO_MODEL_PATH)

# --- メインループ ---
print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: フレーム取得失敗")
        break

    # 1) YOLO推論用に縮小
    small = cv2.resize(frame, YOLO_INPUT_SIZE)
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # 2) ROI内 & 進行方向(0°±10°)ブロックで地面距離取得
    centers, side_ds = [], []
    sw, sh = YOLO_INPUT_SIZE
    fw, fh = frame.shape[1], frame.shape[0]
    sx, sy = fw / sw, fh / sh

    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        theta  = (cx - sw/2) / sw * FOV_DEG

        if abs(theta) <= ROI_ANGLE_DEG:
            centers.append((x1, y1, x2, y2))

        if abs(theta) <= 10:
            y_max = max(y1, y2, cy)
            phi   = ((y_max / sh) - 0.5) * math.pi
            if abs(phi) > 1e-3:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                side_ds.append(Dg)

    # 3) RSSで直進/旋回判定
    direction, rss = 'no', 0.0
    if len(centers) >= 2:
        pts = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2 in centers])
        A   = np.vstack([pts[:,0], np.ones(len(pts))]).T
        m, b= np.linalg.lstsq(A, pts[:,1], rcond=None)[0]
        resid = np.abs(m*pts[:,0] - pts[:,1] + b) / math.hypot(m,1)
        rss   = float((resid**2).sum())
        direction = 'straight' if rss < RSS_THRESH else 'turn'
    history.append(1 if direction == 'straight' else 0)

    # 4) 旋回方向
    side_flag = 'left'
    if centers:
        left_cnt  = sum(1 for x1,_,x2,_ in centers if (x1+x2)/2 < sw/2)
        side_flag = 'right' if len(centers)-left_cnt > left_cnt else 'left'

    # 5) 角度修正
    corrected = False
    if side_ds and centers:
        Dm = sum(side_ds) / len(side_ds)
        corrected = correct_angle_loop(Dm, side_flag)

    # 6) 直進/停止
    if not corrected:
        if sum(history) >= (HISTORY_LEN//2 + 1):
            drive_velocity(SPEED_VALUE, 0)
        else:
            drive_velocity(0, 0)

    # 7) フル解像度にバウンディング描画
    for x1, y1, x2, y2 in boxes.astype(int):
        x1o, y1o = int(x1 * sx), int(y1 * sy)
        x2o, y2o = int(x2 * sx), int(y2 * sy)
        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0,255,0), 2)
    if side_ds:
        cv2.putText(frame, f"D:{Dm:.2f}m", (10, fh-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # 8) 表示用リサイズ（ウィンドウが大きすぎないように）
    disp = frame.copy()
    h_disp, w_disp = disp.shape[:2]
    scale = min(MAX_WIN_W / w_disp, MAX_WIN_H / h_disp, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp,
                          (int(w_disp * scale), int(h_disp * scale)),
                          interpolation=cv2.INTER_AREA)

    cv2.imshow('WHILL Auto Drive', disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 後処理 ---
cap.release()
cv2.destroyAllWindows()
drive_velocity(0, 0)
whill.send_power_off()
print("Program terminated.")