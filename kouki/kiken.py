# corrected_incorze.py

#in_out.pyに対しcurrent_stateでフラグを統一したもの
#不必要な関数、変数は削除した

import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from whill import ComWHILL   # 実機ライブラリ：実行環境に合わせてコメント解除
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque
from datetime import datetime

# --- WHILL 初期化 ---
whill = None
try:
    # 実機がある場合は以下を有効にしてください
    whill = ComWHILL(port='COM3')
    whill.set_power(True)
    time.sleep(1)
    print("WHILL: 初期化処理をスキップ（実機接続がコメントアウトされています）")
except Exception as e:
    print(f"WHILLへの接続に失敗しました: {e}")
    whill = None

# --- パラメータ設定 ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)  # (width, height)
CONF_THRESH       = 0.2
FOV_DEG           = 360.0
ROI_ANGLE_DEG     = 60.0
SIDE_THRESH_DEG   = 10.0
CAM_H             = 1.55
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 40
TURN_SPEED_PCT    = 5  # %（側方補正パーセンテージ）。以前は 2 と小さすぎたため実運用向けに調整しておく
STOP_DISTANCE_M   = 1.5   # ★★★ 追加：インコース旋回を開始する距離(m) ★★★
# ANGLE_MATCH_THRESHOLD  = 10.0

# ★★★ 新しいパラメータを追加 ★★★
POST_SPIKE_DELAY_S = 2.5 # スパイク検出後、何秒間直進を続けるか(s)

# --- Side Block Spike Detection Parameters (for OUT-COURSE) ---
SIDE_COUNT_HISTORY_SIZE = 20  # Number of frames to average for spike detection
# SPIKE_ABS_THRESHOLD     = 5   # (固定値ルールのため、現在未使用)
# SPIKE_REL_FACTOR        = 3.0 # (固定値ルールのため、現在未使用)


# ▼▼▼【重要】この一行を追加してください ▼▼▼
IN_COURSE_DETECT_RANGE_M = 2.2 # インコースと判断し始める距離(m)

POST_TURN_SEARCH_SPEED = 15 # 旋回後に壁を探すときの前進速度 (%)

# RANSAC パラメータ
RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.50
FRONT_TH_DEG      = 15.0

# アライメント（初期整列）パラメータ
DIST_THRESHOLD   = 0.05       # [m]（距離誤差のしきい値）

# --- 移動平均フィルタ ---
WINDOW_SIZE       = 10

# --- ログ・動画の保存設定 ---
LOG_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\final_integrated"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# カメラ準備
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けませんでした。")
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(LOG_DIR, f"run_{timestamp}.mp4"), fourcc, 10.0, (fw, fh))

# --- YOLOモデルのロード ---
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE  # sw: width, sh: height

# ヘルパー関数
def clamp(x, a, b):
    return max(a, min(b, x))

def drive_velocity(f_pct, s_pct=0):
    """
    f_pct: 前進速度（%）
    s_pct: 側方/旋回速度（%）
    WHILL API の送信スケールに依存して変換する。
    """
    # 値をクリップ
    f_pct = clamp(f_pct, -100, 100)
    s_pct = clamp(s_pct, -100, 100)
    if whill:
        try:
            # 実際の送信スケールはデバイス仕様に合わせること
            whill.send_velocity(int(f_pct / 100.0 * 1000), int(s_pct / 100.0 * 1500))
        except Exception as e:
            print(f"WHILL 送信エラー: {e}")
    else:
        pass

# --- 制御フラグの初期化 ---
# aligned   = False
# started   = False
current_state = "WAITING_TO_START"
First_Dg  = None
lower_thr = upper_thr = None
spike_detection_time = None 

print("'s'キーで開始、'q'キーで終了します。")

try:
    course_type = "STRAIGHT"
    turn_direction = "NONE"
    while True:
        # --- ループごとの状態変数をここでまとめて初期化 ---
        stop_flag = False
        # course_type = "STRAIGHT"
        # turn_direction = "NONE"
        direction = 'straight'
        inlier_ratio = 0.0
        avg_Dg = 0.0
        front_block_for_turn = None
        is_spike = False

# while True: の直後から finally: の前までを以下に置き換える

        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break
        disp = frame.copy()

        # 1. キー入力 (常に受付)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 2. 画像処理と物体検出 (常に実行)
        h, w = disp.shape[:2]
        scale_x, scale_y = w / sw, h / sh
        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.empty((0, 4))

        all_info, side_info, roi_info, front_info = [], [], [], []
        if len(boxes) > 0:
            for (x1, y1, x2, y2) in boxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                theta = (cx - sw / 2.0) / sw * FOV_DEG
                y_max = max(y1, y2, cy)
                phi = ((y_max / sh) - 0.5) * math.pi
                Dg = (CAM_H / math.tan(abs(phi)) - ERROR_OFFSET) if abs(math.tan(abs(phi))) > 1e-6 else float('inf')
                rx1, ry1, rx2, ry2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                block_data = (theta, Dg, rx1, ry1, rx2, ry2)
                all_info.append(block_data)
                if abs(theta) <= ROI_ANGLE_DEG: roi_info.append(block_data)
                if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG: side_info.append(block_data)
                if abs(theta) <= FRONT_TH_DEG: front_info.append(block_data)

        # 3. 状態に応じた処理の分岐
        if current_state == "WAITING_TO_START":
            cv2.putText(disp, "Press 's' to start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if key == ord('s'):
                print("開始します。初期アライメントに移行。")
                current_state = "INITIAL_ALIGN"

        elif current_state == "INITIAL_ALIGN":
            # ★★★ ここが手順3の正しい実装です ★★★
            if not side_info or not all_info:
                drive_velocity(0, -TURN_SPEED_PCT)
            else:
                _, D_all, _, _, _, _ = min(all_info, key=lambda x: x[1])
                _, D_side, _, _, _, _ = min(side_info, key=lambda x: x[1])
                error = D_side - D_all
                if error > DIST_THRESHOLD:
                    drive_velocity(0, -TURN_SPEED_PCT)
                elif error < -DIST_THRESHOLD:
                    drive_velocity(0, TURN_SPEED_PCT)
                else:
                    drive_velocity(0, 0)
                    print("初期アライメント完了。壁追従を開始します。")
                    current_state = "WALL_FOLLOWING"

        elif current_state == "WALL_FOLLOWING":
            # ★★★ ここに元の壁追従と旋回判断ロジックを移動 ★★★
            direction = 'straight' # ループごとに初期化
            
            # 基準距離(First_Dg)の設定
            if First_Dg is None:
                if side_info:
                    First_Dg = min(b[1] for b in side_info)
                    print(f"最初の基準距離を設定: {First_Dg:.2f}m")
                    lower_thr = First_Dg - 0.15
                    upper_thr = First_Dg + 0.15
            
            # RANSACによる直線判断
            if roi_info:
                if len(roi_info) >= 5:
                    X = np.array([b[0] for b in roi_info]).reshape(-1, 1)
                    y = np.array([b[1] for b in roi_info])
                    ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX)
                    try:
                        ransac.fit(X, y)
                        inlier_ratio = np.sum(ransac.inlier_mask_) / len(X)
                        if inlier_ratio < INLIER_RATIO_TH: direction = 'turn'
                    except ValueError:
                        direction = 'turn'
                elif len(roi_info) > 0: # ブロックが5個未満でも壁が途切れたと判断
                    direction = 'turn'

            # インコース/アウトコース判断
            front_block_candidate, is_incourse_candidate, is_outcourse_candidate = None, False, False
            if roi_info:
                front_block_candidate_tuple = min(roi_info, key=lambda x: abs(x[0]))
                front_block_candidate = {'theta': front_block_candidate_tuple[0], 'dist': front_block_candidate_tuple[1]}
                if abs(front_block_candidate['theta']) <= FRONT_TH_DEG and front_block_candidate['dist'] <= IN_COURSE_DETECT_RANGE_M:
                    is_incourse_candidate = True
                    course_type = "IN-COURSE"
                    front_block_for_turn = front_block_candidate
            
            if not is_incourse_candidate and direction == 'turn':
                is_outcourse_candidate = True
                course_type = "OUT-COURSE"

            # 旋回トリガー
            if is_incourse_candidate and front_block_for_turn['dist'] <= STOP_DISTANCE_M:
                print("インコース目標を検知。旋回状態へ移行します。")
                drive_velocity(0, 0)
                current_state = "TURNING_IN_COURSE"
            elif is_outcourse_candidate and len(side_info) >= 4: # スパイク条件
                print("アウトコーススパイクを検知。遅延状態へ移行します。")
                spike_detection_time = time.time()
                current_state = "OUT_COURSE_DELAY"
            else:
                # 上記のいずれでもない場合は壁追従走行
                if First_Dg is not None:
                    if side_info:
                        dg = min(side_info, key=lambda b: b[1])[1]
                        side = 0
                        if dg < lower_thr: side = TURN_SPEED_PCT
                        elif dg > upper_thr: side = -TURN_SPEED_PCT
                        drive_velocity(FORWARD_SPEED_PCT, side)
                    else:
                        drive_velocity(POST_TURN_SEARCH_SPEED, 0) # 壁が見えないときはゆっくり前進
                else:
                    drive_velocity(0, 0) # 基準距離がない場合は停止

        # (ここに、今後実装する他の状態のelif節が続きます)
        # メインループのelif節として新設
        elif current_state == "POST_TURN_ALIGN":
            # 初期アライメントとほぼ同じロジック
            if not side_info:
                drive_velocity(0, -TURN_SPEED_PCT)
            else:
                _, D_all, _, _, _, _ = min(all_info, key=lambda x: x[1])
                _, D_side, _, _, _, _ = min(side_info, key=lambda x: x[1])
                error = D_side - D_all
                if error > DIST_THRESHOLD:
                    drive_velocity(0, -TURN_SPEED_PCT)
                elif error < -DIST_THRESHOLD:
                    drive_velocity(0, TURN_SPEED_PCT)
                else:
                    # アライメント完了
                    drive_velocity(0, 0)
                    print("再アライメント完了。壁追従を再開します。")
                    
                    # ★ ここで新しい壁の距離を再設定する
                    if side_info:
                        new_first_dg = min(side_info, key=lambda b: b[1])[1]
                        First_Dg = new_first_dg
                        lower_thr = First_Dg - 0.15
                        upper_thr = First_Dg + 0.15
                        print(f"新しい基準距離: {First_Dg:.2f}m")
                    else:
                        First_Dg = None # 見つからなければリセット
                    
                    # 状態をリセット
                    course_type = "STRAIGHT"
                    turn_direction = "NONE"
                    spike_detection_time = None
                    
                    current_state = "WALL_FOLLOWING" # ★ 壁追従状態に戻る
                    # ▼▼▼【ここから追加】▼▼▼

        elif current_state == "OUT_COURSE_DELAY":
            if spike_detection_time is None:
                # 予期せずこの状態に入った場合は、安全のため壁追従に戻る
                print("エラー: 遅延タイマーが未設定でOUT_COURSE_DELAYに移行しました。")
                current_state = "WALL_FOLLOWING"
            else:
                elapsed = time.time() - spike_detection_time
                if elapsed >= POST_SPIKE_DELAY_S:
                    print(f"遅延完了 ({POST_SPIKE_DELAY_S}秒)。旋回のため停止します。")
                    drive_velocity(0, 0)
                    current_state = "TURNING_OUT_COURSE" # ★アウトコース旋回状態へ移行
                else:
                    # 遅延中は直進
                    drive_velocity(FORWARD_SPEED_PCT, 0)
        
        elif current_state == "TURNING_IN_COURSE":
            # in_out_v1.py のインコース旋回ロジックを移植
            
            if front_block_for_turn is None:
                print("エラー: インコース目標が未設定でTURNING_IN_COURSEに移行しました。90度旋回します。")
                target_angle = 90.0
            else:
                TIME_PER_DEGREE = 0.102; TURN_SPEED_FOR_CALIBRATION = 5
                print("α+δの角度推定と時間制御による旋回を開始します。"); cv2.waitKey(1500)
                ret, stable_frame = cap.read(); disp = stable_frame.copy(); target_angle = 90.0
                if ret:
                    # スケール変数の取得
                    h, w = stable_frame.shape[:2]
                    scale_x, scale_y = w / sw, h / sh
                    
                    small_stable = cv2.resize(stable_frame, (sw, sh)); res_stable = model(small_stable, conf=CONF_THRESH)[0]
                    boxes_stable = res_stable.boxes.xyxy.cpu().numpy(); front_blocks, side_blocks = [], []
                    if len(boxes_stable) > 0:
                        for x1,y1,x2,y2 in boxes_stable:
                            cx_s, cy_s = (x1+x2)/2.0, (y1+y2)/2.0; theta_s = (cx_s - sw/2.0)/sw * FOV_DEG
                            rx1, ry1, rx2, ry2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                            center_point = np.array([(rx1+rx2)/2, (ry1+ry2)/2])
                            block_info = {'box': (rx1, ry1, rx2, ry2), 'center': center_point}
                            if abs(theta_s) <= FRONT_TH_DEG: front_blocks.append(block_info)
                            is_valid_side = (turn_direction == "RIGHT TURN" and theta_s < 0) or \
                                          (turn_direction == "LEFT TURN" and theta_s > 0)
                            if is_valid_side: side_blocks.append(block_info)
                    angle_alpha, angle_delta = 0.0, 0.0
                    if front_blocks and side_blocks:
                        sorted_front_blocks = sorted(front_blocks, key=lambda b: b['center'][1], reverse=True)
                        found_valid_angle = False
                        for b_candidate in sorted_front_blocks:
                            point_B = b_candidate['center']
                            c_candidate = min(side_blocks, key=lambda c: np.linalg.norm(c['center'] - point_B))
                            point_C = c_candidate['center']
                            if np.linalg.norm(point_B - point_C) < 50: continue
                            camera_A = np.array([w / 2, h]); vec_AB = point_B - camera_A; vec_AC = point_C - camera_A
                            vec_CB = point_B - point_C; vec_CA = camera_A - point_C
                            dot_d = np.dot(vec_AB, vec_AC); mag_AB = np.linalg.norm(vec_AB); mag_AC = np.linalg.norm(vec_AC)
                            angle_delta = np.degrees(np.arccos(clamp(dot_d / (mag_AB * mag_AC), -1.0, 1.0))) if mag_AB > 0 and mag_AC > 0 else 0
                            dot_a = np.dot(vec_CB, vec_CA); mag_CB = np.linalg.norm(vec_CB); mag_CA = np.linalg.norm(vec_CA)
                            angle_alpha = np.degrees(np.arccos(clamp(dot_a / (mag_CB * mag_CA), -1.0, 1.0))) if mag_CB > 0 and mag_CA > 0 else 0
                            total_angle = angle_alpha + angle_delta
                            if total_angle > 10.0:
                                target_angle = total_angle; found_valid_angle = True
                                print(f"有効な角度を検出しました: B={point_B.astype(int)}, C={point_C.astype(int)}"); break
                        if not found_valid_angle:
                            print("警告: 全ての候補ペアで有効な角度を推定できませんでした。90度で旋回します。")
                            target_angle = 90.0; angle_alpha, angle_delta = 0, 0
                        cv2.putText(disp, f"Alpha (a): {angle_alpha:.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(disp, f"Delta (d): {angle_delta:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(disp, f"TARGET (a+d): {target_angle:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else: print("角度推定に必要なブロックが見つかりませんでした。デフォルトの90度で旋回します。")
                    cv2.imshow('Control', disp); cv2.waitKey(2000)

            # 旋回実行
            turn_duration = target_angle * TIME_PER_DEGREE
            start_time = time.time()
            while time.time() - start_time < turn_duration:
                ret, turn_frame = cap.read()
                if not ret: break
                cv2.putText(turn_frame, f"Turning... Target: {target_angle:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Control', turn_frame); out.write(turn_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): print("手動で旋回を中断しました。"); break
                drive_velocity(0, TURN_SPEED_FOR_CALIBRATION if turn_direction == "RIGHT TURN" else -TURN_SPEED_FOR_CALIBRATION)
            drive_velocity(0, 0); print("旋回完了。"); cv2.waitKey(1500)
            
            # ★旋回後のアライメント状態へ移行
            print("インコース旋回完了。再アライメントへ移行します。")
            current_state = "POST_TURN_ALIGN"

        elif current_state == "TURNING_OUT_COURSE":
            # in_out_v1.py のアウトコース旋回ロジックを移植
            TIME_PER_DEGREE = 0.122; TURN_SPEED_FOR_CALIBRATION = 5; TARGET_ANGLE = 90.0
            print(f"アウトコース：固定{TARGET_ANGLE}度の旋回を開始します。"); cv2.waitKey(1500)
            turn_duration = TARGET_ANGLE * TIME_PER_DEGREE
            start_time = time.time()
            while time.time() - start_time < turn_duration:
                ret, turn_frame = cap.read()
                if not ret: break
                cv2.putText(turn_frame, f"Turning OUT-COURSE... Target: {TARGET_ANGLE:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Control', turn_frame); out.write(turn_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                drive_velocity(0, TURN_SPEED_FOR_CALIBRATION if turn_direction == "RIGHT TURN" else -TURN_SPEED_FOR_CALIBRATION)
            drive_velocity(0, 0); print("旋回完了。"); cv2.waitKey(1500)

            print("新しい壁が見つかるまで、ゆっくり前進します...");
            search_start_time = time.time()
            found_wall = False
            
            stable_wall_counter = 0
            STABLE_FRAME_THRESHOLD = 10 # in_out_v1.py から持ってくる

            # スケール変数の取得 (cap.getは重いので初期値fw, fhを使う)
            h, w = fh, fw
            scale_x, scale_y = w / sw, h / sh

            while True:
                if time.time() - search_start_time > 60.0: 
                    print("タイムアウト: 新しい壁が見つかりませんでした。")
                    break
                
                drive_velocity(POST_TURN_SEARCH_SPEED, 0); ret, search_frame = cap.read()
                if not ret: break
                
                small_search = cv2.resize(search_frame, (sw, sh)); res_search = model(small_search, conf=CONF_THRESH)[0]
                
                boxes_search = res_search.boxes.xyxy.cpu().numpy(); side_blocks_new = []
                if len(boxes_search) > 0:
                    for x1,y1,x2,y2 in boxes_search:
                        cx_n, cy_n = (x1+x2)/2.0, (y1+y2)/2.0; theta_n = (cx_n - sw/2.0)/sw * FOV_DEG
                        
                        is_new_side = (turn_direction == "RIGHT TURN" and theta_n > 0) or \
                                      (turn_direction == "LEFT TURN" and theta_n < 0)
                        
                        if is_new_side and abs(abs(theta_n)-90.0) <= SIDE_THRESH_DEG:
                            y_max_n = max(y1,y2,cy_n); phi_n = ((y_max_n / sh) - 0.5) * math.pi
                            Dg_n = CAM_H / math.tan(abs(phi_n)) - ERROR_OFFSET if abs(math.tan(abs(phi_n))) > 1e-6 else float('inf')
                            side_blocks_new.append({'dist': Dg_n})
                
                num_side_blocks = len(side_blocks_new)
                is_stable_condition = (1 <= num_side_blocks <= 2)

                if is_stable_condition:
                    stable_wall_counter += 1
                else:
                    stable_wall_counter = 0

                # (見える化のコードは省略可。in_out_v1.pyから移植してもOK)

                cv2.putText(search_frame, f"Searching... Stable Count: {stable_wall_counter}/{STABLE_FRAME_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Control', search_frame); out.write(search_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                if stable_wall_counter >= STABLE_FRAME_THRESHOLD:
                    drive_velocity(0, 0)
                    print(f"安定した壁を{STABLE_FRAME_THRESHOLD}フレーム連続で確認。探索を終了します。")
                    found_wall = True
                    break

            # ★旋回後のアライメント状態へ移行
            print("アウトコース探索完了。再アライメントへ移行します。")
            current_state = "POST_TURN_ALIGN"

        # ▲▲▲【ここまで追加】▲▲▲

        # (この次に existing elif current_state == "POST_TURN_ALIGN": が来る)


                    
        # 4. 描画処理 (常に実行)
        # --- 映像保存 & 表示 ---
        status_font = cv2.FONT_HERSHEY_SIMPLEX
        status_scale = 0.7
        status_thickness = 2
        info_color = (255, 255, 0)
        alert_color = (0, 0, 255)

        # 左上に表示するテキスト群
        cv2.putText(disp, f"State: {current_state}", (10, 30), status_font, status_scale, (0, 255, 255), status_thickness)
        cv2.putText(disp, f"Course: {course_type}", (10, 60), status_font, status_scale, info_color, status_thickness)
        cv2.putText(disp, f"Path: {direction.upper()}", (10, 90), status_font, status_scale, info_color, status_thickness)
        if is_spike:
            cv2.putText(disp, "SPIKE DETECTED!", (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.8, alert_color, status_thickness)

        # 映像の書き込みと表示は最後に1回
        out.write(disp)
        cv2.imshow('Control', disp)

finally:
    # 終了処理（必ず実行）
    print("プログラムを終了します...")
    drive_velocity(0, 0)
    if whill:
        try: whill.set_power(False)
        except Exception as e: print(f"WHILL 電源オフエラー: {e}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("処理が完了し、動画が保存されました。")