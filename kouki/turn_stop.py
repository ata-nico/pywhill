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
# WHILLに接続し、電源をONにします。
# 接続に失敗した場合は、'COM3'の部分を適切なポートに変更してください。
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID        = 0      # 使用するカメラのID
YOLO_MODEL_PATH  = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt" # YOLOモデルのパス
YOLO_INPUT_SIZE  = (640, 480) # YOLOモデルへの入力画像サイズ
CONF_THRESH      = 0.2    # 物体検出の信頼度しきい値
FOV_DEG          = 360    # カメラの水平視野角 [deg]
ROI_ANGLE_DEG    = 60     # 制御対象とする前方の関心領域 (Region of Interest) の角度 [deg]
SIDE_THRESH_DEG  = 10     # 真横を判定するための角度の閾値 [deg]
CAM_H            = 1.55   # カメラの地面からの高さ [m]
ERROR_OFFSET     = 0.0    # 距離計算のオフセット
FORWARD_SPEED_PCT= 40     # 前進速度 (%)
TURN_SPEED_PCT   = 1      # 旋回速度 (%)
LOOP_INTERVAL    = 0.1    # 制御ループの間隔 [s]

# RANSAC (直線検出) パラメータ
RANSAC_THRESH_PX = 3.0    # RANSACのインライア（直線上の点）と見なす距離の閾値 [pixel]
INLIER_RATIO_TH  = 0.55   # 直線と判定するためのインライアの割合のしきい値

# Alignment (初期整列) 制御パラメータ
Kp              = 50      # 比例制御ゲイン
DIST_THRESHOLD  = 0.05    # 整列完了と見なす距離の誤差 [m]

# --- 移動平均フィルタ (ウィンドウサイズを10に変更) ---
WINDOW_SIZE       = 10
dg_history        = deque(maxlen=WINDOW_SIZE) # 距離の履歴
direction_history = deque(maxlen=WINDOW_SIZE) # 直進/カーブ判定の履歴

# --- ログ保存設定 ---
LOG_DIR   = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\smooth"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- 動画出力設定 ---
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
    """WHILLに速度指令を送信する関数"""
    whill.send_velocity(int(f_pct/100*1000), int(s_pct/100*1500))

# --- 制御フラグ初期化 ---
aligned   = False # 整列が完了したかどうかのフラグ
started   = False # 走行が開始されたかどうかのフラグ
First_Dg  = None  # 基準となる初期距離
lower_thr = upper_thr = None # 距離維持のための閾値

print("Press 's' to start, 'q' to quit.")

# --- メインループ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    disp = frame.copy()

    # 開始キー入力待ち
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

    # YOLOによる物体検出
    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    
    # 検出した各ブロックの距離と角度を計算
    all_info = []
    side_info = []
    roi_info  = []
    if len(boxes) > 0:
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

    # --- 描画処理 ---
    # ROIの枠 (白)
    x_roi1 = int((0.5 - ROI_ANGLE_DEG/FOV_DEG) * fw)
    x_roi2 = int((0.5 + ROI_ANGLE_DEG/FOV_DEG) * fw)
    cv2.rectangle(disp, (x_roi1, 0), (x_roi2, fh), (255,255,255), 2)

    # ROI内の全ブロックを薄い白で描画
    for _,_,x1,y1,x2,y2 in roi_info:
        cv2.rectangle(disp, (int(x1*fw/sw), int(y1*fh/sh)),
                      (int(x2*fw/sw), int(y2*fh/sh)), (255,255,255), 1)

    # 注目ブロックの特定と描画
    closest_block_dist = float('inf')
    
    # 全体で最も近いブロック (青) - これは常に計算・表示
    if all_info:
        _, closest_block_dist, ax1, ay1, ax2, ay2 = min(all_info, key=lambda x: x[1])
        cv2.rectangle(disp, (int(ax1*fw/sw), int(ay1*fh/sh)),
                      (int(ax2*fw/sw), int(ay2*fh/sh)), (255,0,0), 2)

    # 真横で最も近いブロック (緑)
    if side_info:
        _,_,sx1,sy1,sx2,sy2 = min(side_info, key=lambda x: x[1])
        cv2.rectangle(disp, (int(sx1*fw/sw), int(sy1*fh/sh)),
                      (int(sx2*fw/sw), int(sy2*fh/sh)), (0,255,0), 2)

    # 最短距離情報のテキスト表示
    cv2.putText(disp, f"Closest: {closest_block_dist:.2f}m", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # 青字

    # --- 制御ロジック ---
    if len(boxes) == 0:
        drive_velocity(0,0)
        cv2.putText(disp, "No detection", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        out.write(disp); cv2.imshow('Control', disp)
        continue

    # フェーズ1: アライメント
    if not aligned:
        if not side_info:
            drive_velocity(0, -TURN_SPEED_PCT)
            out.write(disp); cv2.imshow('Control', disp)
            continue
        t_all,D_all,*_ = min(all_info, key=lambda x:x[1])
        t_side,D_side,*_ = min(side_info, key=lambda x:x[1])
        error = D_side - D_all
        status = ""
        if t_all>0 and error> DIST_THRESHOLD:
            drive_velocity(0, -int(Kp*error)); status="Align: Turn Left"
        elif t_all>0 and error< -DIST_THRESHOLD:
            drive_velocity(0, TURN_SPEED_PCT); status="Align: Turn Right"
        elif t_all<0 and error> DIST_THRESHOLD:
            drive_velocity(0, int(Kp*error)); status="Align: Turn Right"
        elif t_all<0 and error< -DIST_THRESHOLD:
            drive_velocity(0, -TURN_SPEED_PCT); status="Align: Turn Left"
        else:
            drive_velocity(0,0); status="Aligned"; aligned=True
        cv2.putText(disp, status, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        # フェーズ2: メイン制御
        dg_history.append(closest_block_dist)
        avg_Dg = np.mean(dg_history)
        if First_Dg is None:
            First_Dg = closest_block_dist
            lower_thr = First_Dg - 0.15
            upper_thr = First_Dg + 0.15
            print(f"First_Dg={First_Dg:.2f}, Thr=[{lower_thr:.2f},{upper_thr:.2f}]")

        # RANSACによる直線/カーブ判定
        direction = 'straight'
        inlier_ratio = 0.0
        if len(roi_info) >= 2:
            pts = np.array([[(x1+x2)/2, (y1+y2)/2] for _,_,x1,y1,x2,y2 in roi_info])
            X = pts[:,0].reshape(-1,1); Y = pts[:,1]
            ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
            ransac.fit(X,Y)
            inliers = ransac.inlier_mask_
            inlier_ratio = inliers.sum() / len(inliers)
            direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'
        
        direction_history.append(direction)

        # ★★★ 正面ブロックの処理（turnの時のみ）★★★
        front_block = None
        front_block_dist = float('inf')
        if direction == 'turn':
            if roi_info:
                front_block = min(roi_info, key=lambda x: abs(x[0]))
                _, front_block_dist, fx1, fy1, fx2, fy2 = front_block
                # 黄色い枠で描画
                cv2.rectangle(disp, (int(fx1*fw/sw), int(fy1*fh/sh)),
                              (int(fx2*fw/sw), int(fy2*fh/sh)), (0,255,255), 2)
                # 距離をテキストで表示
                cv2.putText(disp, f"Front: {front_block_dist:.2f}m", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 新しい停止ロジック
        if direction_history.count('turn') >= WINDOW_SIZE / 2:
            if front_block is not None: # front_blockはturnの時しか設定されない
                if front_block_dist <= 1.5:
                    print(f"Front block detected at {front_block_dist:.2f}m. Stopping.")
                    drive_velocity(0, 0)
                    cv2.putText(disp, f"STOP! Front block too close: {front_block_dist:.2f}m", (10, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    out.write(disp)
                    cv2.imshow('Control', disp)
                    cv2.waitKey(1)
                    break
        
        forward, side = 0,0
        if avg_Dg < lower_thr:
            forward=FORWARD_SPEED_PCT; side=-TURN_SPEED_PCT if direction=='turn' else TURN_SPEED_PCT
        elif avg_Dg > upper_thr:
            forward=FORWARD_SPEED_PCT; side= TURN_SPEED_PCT if direction=='turn' else -TURN_SPEED_PCT
        elif direction_history.count('turn') < WINDOW_SIZE:
            forward=FORWARD_SPEED_PCT
            
        drive_velocity(forward, side)
        
        # 表示テキストの変更
        direction_text = f"{direction.upper()} ({inlier_ratio:.3f})"
        cv2.putText(disp, f"F:{forward} S:{side} {direction_text} Turns:{direction_history.count('turn')}", 
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out.write(disp); cv2.imshow('Control', disp)

# --- 終了処理 ---
print("Exiting program...")
drive_velocity(0,0)
whill.set_power(False)
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved.")

