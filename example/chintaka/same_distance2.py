#!/usr/bin/env python3
import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque
from datetime import datetime

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID        = 0
YOLO_MODEL_PATH  = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
YOLO_INPUT_SIZE  = (640, 480)
CONF_THRESH      = 0.2
FOV_DEG          = 360
ROI_ANGLE_DEG    = 60      # 前方ROI角度 [deg]
SIDE_THRESH_DEG  = 10      # 真横検出用閾値 [deg]
CAM_H            = 1.55
ERROR_OFFSET     = 0.0
FORWARD_SPEED_PCT= 40
TURN_SPEED_PCT   = 1
LOOP_INTERVAL    = 0.1

RANSAC_THRESH_PX = 5.0
INLIER_RATIO_TH  = 0.53

# Alignment 制御パラメータ
Kp              = 50
DIST_THRESHOLD  = 0.05

# --- 移動平均フィルタ ---
WINDOW_SIZE       = 5
dg_history        = deque(maxlen=WINDOW_SIZE)
direction_history= deque(maxlen=WINDOW_SIZE)

# --- 連続非直進カウンタ ---
non_straight_count = 0

# --- ログ記録 ---
times  = []
errors = []
dt     = LOOP_INTERVAL

# --- ログ保存設定 ---
LOG_DIR   = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\smooth"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- 動画出力 ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened(): raise RuntimeError(f"Camera {CAMERA_ID} open error")
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(LOG_DIR, f"run_{timestamp}.mp4"), fourcc, 10.0, (fw, fh))

# --- YOLOモデルロード ---
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

# --- ヘルパー関数 ---
def drive_velocity(f_pct, s_pct=0):
    whill.send_velocity(int(f_pct/100*1000), int(s_pct/100*1500))

# --- 制御フラグ初期化 ---
aligned   = False
started   = False
First_Dg  = None
lower_thr = upper_thr = None

print("Press 's' to start, 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    disp = frame.copy()

    # 開始キー待ち
    key = cv2.waitKey(1) & 0xFF
    if not started:
        cv2.putText(disp, "Press 's' to start", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Control', disp)
        if key == ord('s'):
            started = True
        elif key == ord('q'):
            break
        continue
    if key == ord('q'):
        break

    # YOLO推論
    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        drive_velocity(0,0)
        error = 0.0
        times.append(time.time()-t0)
        errors.append(error)
        cv2.putText(disp, "No detection", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        out.write(disp); cv2.imshow('Control', disp)
        continue

    # 距離計算と角度分類
    all_info = []       # (theta, Dg, x1, y1, x2, y2)
    side_info = []
    roi_info  = []
    for x1,y1,x2,y2 in boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        theta = (cx - sw/2)/sw * FOV_DEG
        y_max = max(y1,y2,cy)
        phi = ((y_max/sh)-0.5)*math.pi
        Dg = float('inf') if abs(phi)<1e-3 else CAM_H/math.tan(abs(phi)) - ERROR_OFFSET
        all_info.append((theta, Dg, x1,y1,x2,y2))
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((theta, Dg, x1,y1,x2,y2))
        if abs(abs(theta)-90.0) <= SIDE_THRESH_DEG:
            side_info.append((theta, Dg, x1,y1,x2,y2))

    # 描画：ROI 枠 (白)
    x_roi1 = int((0.5 - ROI_ANGLE_DEG/FOV_DEG) * fw)
    x_roi2 = int((0.5 + ROI_ANGLE_DEG/FOV_DEG) * fw)
    cv2.rectangle(disp, (x_roi1, 0), (x_roi2, fh), (255,255,255), 2)
    # 描画：ROI内ブロック(白)
    for _,_,x1,y1,x2,y2 in roi_info:
        cv2.rectangle(disp, (int(x1*fw/sw), int(y1*fh/sh)),
                      (int(x2*fw/sw), int(y2*fh/sh)), (255,255,255), 1)
    # 描画：真横最短(緑)
    if side_info:
        _,_,sx1,sy1,sx2,sy2 = min(side_info, key=lambda x: x[1])
        cv2.rectangle(disp, (int(sx1*fw/sw), int(sy1*fh/sh)),
                      (int(sx2*fw/sw), int(sy2*fh/sh)), (0,255,0), 2)
    # 描画：全ブロック最短(青)
    if all_info:
        _,_,ax1,ay1,ax2,ay2 = min(all_info, key=lambda x: x[1])
        cv2.rectangle(disp, (int(ax1*fw/sw), int(ay1*fh/sh)),
                      (int(ax2*fw/sw), int(ay2*fh/sh)), (255,0,0), 2)

    # アライメントフェーズ
    if not aligned:
        if not side_info:
            # サイド探索のため緩やかに右回転 (逆に設定)
            drive_velocity(0, -TURN_SPEED_PCT)
            error = 0.0
            times.append(time.time()-t0); errors.append(error)
            out.write(disp); cv2.imshow('Control', disp)
            continue
        t_all,D_all,*_ = min(all_info, key=lambda x:x[1])
        t_side,D_side,*_ = min(side_info, key=lambda x:x[1])
        error = D_side - D_all
        # 5パターン比例制御 (回転方向を逆に)
        if t_all>0 and error> DIST_THRESHOLD:
            # 右側遠い→左回転
            drive_velocity(0, -int(Kp*error)); status="右&遠→左回転"
        elif t_all>0 and error< -DIST_THRESHOLD:
            # 右側近い→右離反(右回)
            drive_velocity(0, TURN_SPEED_PCT); status="右&近→右回転"
        elif t_all<0 and error> DIST_THRESHOLD:
            # 左側遠い→右回転
            drive_velocity(0, int(Kp*error)); status="左&遠→右回転"
        elif t_all<0 and error< -DIST_THRESHOLD:
            # 左側近い→左離反
            drive_velocity(0, -TURN_SPEED_PCT); status="左&近→左回転"
        else:
            drive_velocity(0,0); status="整列完了"; aligned=True
        cv2.putText(disp, status, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        # メイン制御フェーズ：ROI内ブロックで直線判定
        _, D_all, *_= min(all_info, key=lambda x:x[1])
        dg_history.append(D_all)
        avg_Dg = np.mean(dg_history)
        if First_Dg is None:
            First_Dg = D_all
            lower_thr = First_Dg - 0.15
            upper_thr = First_Dg + 0.15
            print(f"First_Dg={First_Dg:.2f}, Thr=[{lower_thr:.2f},{upper_thr:.2f}]")
        direction = 'straight'
        if len(roi_info) >= 2:
            pts = np.array([[(x1+x2)/2, (y1+y2)/2] for _,_,x1,y1,x2,y2 in roi_info])
            X = pts[:,0].reshape(-1,1); Y = pts[:,1]
            ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
            ransac.fit(X,Y)
            inliers = ransac.inlier_mask_
            direction = 'straight' if inliers.sum()/len(inliers) >= INLIER_RATIO_TH else 'turn'
        direction_history.append(direction)
        non_straight_count = non_straight_count+1 if direction!='straight' else 0
        if non_straight_count>=10:
            print("Non-straight 10連, exit")
            break
        forward, side = 0,0
        if avg_Dg < lower_thr:
            forward=FORWARD_SPEED_PCT; side=-TURN_SPEED_PCT if direction=='turn' else TURN_SPEED_PCT
        elif avg_Dg > upper_thr:
            forward=FORWARD_SPEED_PCT; side= TURN_SPEED_PCT if direction=='turn' else -TURN_SPEED_PCT
        elif direction_history.count('turn') < WINDOW_SIZE:
            forward=FORWARD_SPEED_PCT
        drive_velocity(forward, side)
        cv2.putText(disp, f"F:{forward} S:{side}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        error = abs(avg_Dg - First_Dg)
    times.append(time.time()-t0); errors.append(error)
    out.write(disp); cv2.imshow('Control', disp)

# 終了処理
drive_velocity(0,0)
whill.set_power(False)
cap.release(); out.release(); cv2.destroyAllWindows()

# ログCSV保存
n = min(len(times), len(errors))
if n>0:
    arr = np.column_stack((times[:n], errors[:n]))
    np.savetxt(os.path.join(LOG_DIR, f"log_{timestamp}.csv"), arr, delimiter=',', header='time,error', comments='')
    print("Logs saved.")
else:
    print("No log data.")