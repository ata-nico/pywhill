#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque
import threading

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans18\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
CAM_H             = 1.55
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 60
TURN_SPEED_PCT    = 3
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.55
JOYSTICK_DEADZONE = 5  # %

# 出力動画設定
OUTPUT_DIR  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course_with_joystick.mp4"
FPS         = 20.0

# --- 状態管理 ---
direction_history = deque(maxlen=20)
started = False
First_Dg = None                   # 最初の距離を保持
joystick_override = False         # ジョイスティック入力後の自動走行停止フラグ
whill.last_joystick = (0, 0)      # 最後のジョイスティック入力

# --- ヘルパー関数 ---
def drive_velocity(front_pct, side_pct=0):
    whill.send_velocity(int(front_pct/100*1000), int(side_pct/100*1500))

# ジョイスティック入力を監視 (外部呼び出しに対応)
def joystick_listener():
    while True:
        whill.send_joystick(int(10), int(0))
        time.sleep(0.05)

# ジョイスティックスレッド起動
t = threading.Thread(target=joystick_listener, daemon=True)
t.start()

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

    # 1) ジョイスティック優先チェック
    x_pct, y_pct = whill.last_joystick
    if abs(x_pct) > JOYSTICK_DEADZONE or abs(y_pct) > JOYSTICK_DEADZONE:
        joystick_override = True
        whill.send_joystick(int(x_pct), int(y_pct))
        time.sleep(LOOP_INTERVAL)
        continue

    # 2) キー入力待ち／自動走行再開チェック
    ret, frame = cap.read()
    if not ret:
        break
    disp = frame.copy()
    if not started or joystick_override:
        cv2.putText(disp, "Press 's' to start, joystick stops auto", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow('WHILL Control', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            started = True
            joystick_override = False
            First_Dg = None
            direction_history.clear()
            print("[INFO] Autonomous resumed")
        elif key == ord('q'):
            break
        continue

    # 3) 自動走行処理
    # YOLO 推論
    small = cv2.resize(frame, (sw, sh))
    res = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # ROI設定
    fh_, fw_ = frame.shape[:2]
    sx, sy = fw_/sw, fh_/sh
    x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * fw_)
    x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * fw_)

    # ROI内検出
    roi_info = []
    for x1,y1,x2,y2 in boxes:
        cx = (x1+x2)/2
        theta = (cx - sw/2)/sw * FOV_DEG
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

    # 距離計算
    side_info = []
    for x1,y1,x2,y2 in boxes:
        y_max = max(y1,y2,(y1+y2)/2)
        phi = ((y_max/sh)-0.5) * math.pi
        Dg = float('inf') if abs(phi)<1e-3 else max(0.0, CAM_H/math.tan(abs(phi)) - ERROR_OFFSET)
        side_info.append((int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy), Dg))

    target = min(side_info, key=lambda t:t[4], default=None)
    min_Dg = target[4] if target else None
    if target and First_Dg is None:
        First_Dg = min_Dg
        print(f"[INFO] First_Dg set to {First_Dg:.2f}m")

    # 偏り判定
    roi_mid = (x1_roi + x2_roi)/2
    left_n = sum(1 for (x1o,_,x2o,_) in roi_info if (x1o+x2o)/2 < roi_mid)
    right_n= len(roi_info)-left_n
    majority = 'Left' if left_n>right_n else 'Right' if right_n>left_n else 'Equal'

    # RANSAC判定
    direction, inlier_ratio, masks = 'no detect',0.0,None
    if len(roi_info)>=2:
        pts = np.array([[(x1+x2)/2,(y1+y2)/2] for x1,y1,x2,y2 in roi_info])
        X,Y = pts[:,0].reshape(-1,1),pts[:,1]
        ransac=RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
        ransac.fit(X,Y)
        masks=ransac.inlier_mask_
        inlier_ratio=masks.sum()/len(masks)
        direction='straight' if inlier_ratio>=INLIER_RATIO_TH else 'turn'
    direction_history.append(direction)

    # 制御
    forward=0; side=0
    if First_Dg is not None and min_Dg is not None:
        lo,hi = First_Dg-0.2,First_Dg+0.2
        if min_Dg<lo:
            forward=FORWARD_SPEED_PCT; side = -TURN_SPEED_PCT if majority=='Right' else TURN_SPEED_PCT
        elif min_Dg>hi:
            forward=FORWARD_SPEED_PCT; side = TURN_SPEED_PCT if majority=='Right' else -TURN_SPEED_PCT
        else:
            if direction_history.count('turn')<10 and roi_info and inlier_ratio>=INLIER_RATIO_TH:
                forward=FORWARD_SPEED_PCT
        cv2.putText(frame,f"Ref:{First_Dg:.2f}(±0.20) Dg:{min_Dg:.2f}",(20,70),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    drive_velocity(forward, side)

    # 描画・表示
    cv2.rectangle(frame,(x1_roi,0),(x2_roi,fh_),(255,0,0),2)
    if masks is not None:
        for i,(x1o,y1o,x2o,y2o) in enumerate(roi_info):
            c=(0,255,0) if masks[i] else (0,165,255)
            cv2.rectangle(frame,(x1o,y1o),(x2o,y2o),c,2)
    if target:
        tx1,ty1,tx2,ty2,_=target
        cv2.rectangle(frame,(tx1,ty1),(tx2,ty2),(0,0,255),2)
        cv2.putText(frame,f"Dg:{min_Dg:.2f}m",(tx1,ty1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,f"Dir:{direction} Inl:{inlier_ratio:.2f} Side:{majority}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    out.write(frame); cv2.imshow('WHILL Control',frame)

    if cv2.waitKey(1)&0xFF==ord('q'): break

    dt=time.time()-t0
    if dt<LOOP_INTERVAL: time.sleep(LOOP_INTERVAL-dt)

# 終了処理
execute_velocity(0,0)
whill.set_power(False)
cap.release(); out.release(); cv2.destroyAllWindows()
