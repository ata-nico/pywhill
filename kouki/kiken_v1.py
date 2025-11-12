
#kiken.pyの各状態そのものを関数とした

"""
メインループの中を、画像処理・キー入力・物体検出・ブロック分類・状態分岐・描画・表示、という全体の流れだけを管理に変更
RANSACや角度計算などの各処理を関数とした
"""

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
STOP_DISTANCE_M   = 1.5   
# ANGLE_MATCH_THRESHOLD  = 10.0
POST_SPIKE_DELAY_S = 2.5 # スパイク検出後、何秒間直進を続けるか(s)
# --- Side Block Spike Detection Parameters (for OUT-COURSE) ---
SIDE_COUNT_HISTORY_SIZE = 20  # Number of frames to average for spike detection
# SPIKE_ABS_THRESHOLD     = 5   # (固定値ルールのため、現在未使用)
# SPIKE_REL_FACTOR        = 3.0 # (固定値ルールのため、現在未使用)
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

# ヘルパー関数　　値の制限＞速度が範囲を超えないように制限するための関数
def clamp(x, a, b):
    return max(a, min(b, x))

#whillへの動力命令　車椅子を動かす処理をすべてここで行う
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


#ヘルパー関数の定義

#壁との平衡調整
def perform_alignment_step(all_info, side_info):
    """
    壁とのアライメント（平行調整）を1ステップ実行する。
    戻り値: 'ALIGNING' (調整中), 'ALIGNED' (調整完了), 'SEARCHING' (対象なし)
    """
    if not side_info or not all_info:
        drive_velocity(0, -TURN_SPEED_PCT) # 対象が見えないので探す
        return 'SEARCHING'
    
    _, D_all, _, _, _, _ = min(all_info, key=lambda x: x[1])
    _, D_side, _, _, _, _ = min(side_info, key=lambda x: x[1])
    error = D_side - D_all

    if error > DIST_THRESHOLD:
        drive_velocity(0, -TURN_SPEED_PCT)
        return 'ALIGNING'
    elif error < -DIST_THRESHOLD:
        drive_velocity(0, TURN_SPEED_PCT)
        return 'ALIGNING'
    else:
        drive_velocity(0, 0)
        return 'ALIGNED' # 完了

#点字ブロックの検出
def classify_blocks(boxes, sw, sh, scale_x, scale_y):
    """
    YOLOの検出結果(boxes)を4つのカテゴリに分類する。
    """
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
    return all_info, side_info, roi_info, front_info

#情報の描画
def draw_overlay(disp, current_state, course_type, direction, is_spike):
    """
    現在の状態を画面に描画する。
    """
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

# 状態ハンドラ関数の定義　具体的な処理

#キー入力
def handle_state_waiting(key):
    """
    [WAITING_TO_START] 状態の処理。
    's'キーが押されたら状態を遷移する。
    戻り値: new_state (str)
    """
    if key == ord('s'):
        print("開始します。初期アライメントに移行。")
        return "INITIAL_ALIGN"
    return "WAITING_TO_START"

#初期状態の処理
def handle_state_initial_align(all_info, side_info):
    """
    [INITIAL_ALIGN] 状態の処理。
    アライメントが完了したら状態を遷移する。
    戻り値: new_state (str)
    """
    status = perform_alignment_step(all_info, side_info)
    if status == 'ALIGNED':
        print("初期アライメント完了。壁追従を開始します。")
        return "WALL_FOLLOWING"
    return "INITIAL_ALIGN"

#RANSACで直線性を評価、インコース・アウトコースのチェックを行う
def handle_state_wall_following(roi_info, side_info, First_Dg, lower_thr, upper_thr):
    """
    [WALL_FOLLOWING] 状態の処理。
    壁追従走行と、旋回判断を行う。
    戻り値: (new_state, new_course_type, new_turn_direction, new_front_block_for_turn, new_spike_detection_time, new_is_spike, new_direction)
    """
    # この関数内で使う変数の初期化
    direction = 'straight'
    is_spike = False
    new_state = "WALL_FOLLOWING"
    new_course_type = "STRAIGHT"
    new_turn_direction = "NONE"
    new_front_block_for_turn = None
    new_spike_detection_time = None

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
        elif len(roi_info) > 0:
            direction = 'turn'

    # インコース/アウトコース判断
    front_block_candidate, is_incourse_candidate, is_outcourse_candidate = None, False, False
    if roi_info:
        front_block_candidate_tuple = min(roi_info, key=lambda x: abs(x[0]))
        front_block_candidate = {'theta': front_block_candidate_tuple[0], 'dist': front_block_candidate_tuple[1]}
        if abs(front_block_candidate['theta']) <= FRONT_TH_DEG and front_block_candidate['dist'] <= IN_COURSE_DETECT_RANGE_M:
            is_incourse_candidate = True
            new_course_type = "IN-COURSE"
            new_front_block_for_turn = front_block_candidate

    if not is_incourse_candidate and direction == 'turn':
        is_outcourse_candidate = True
        new_course_type = "OUT-COURSE"

    # 旋回トリガー
    if is_incourse_candidate and new_front_block_for_turn['dist'] <= STOP_DISTANCE_M:
        print("インコース目標を検知。旋回状態へ移行します。")
        drive_velocity(0, 0)
        new_state = "TURNING_IN_COURSE"
    elif is_outcourse_candidate and len(side_info) >= 4: # スパイク条件
        print("アウトコーススパイクを検知。遅延状態へ移行します。")
        is_spike = True # 描画フラグ
        new_spike_detection_time = time.time()
        new_state = "OUT_COURSE_DELAY"
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

    # 旋回方向の決定 (状態遷移する/しないに関わらず更新)
    if (is_incourse_candidate or is_outcourse_candidate) and side_info:
        avg_side_theta = np.mean([b[0] for b in side_info])
        is_wall_on_left = (avg_side_theta < 0)
        if new_course_type == "IN-COURSE": new_turn_direction = "RIGHT TURN" if is_wall_on_left else "LEFT TURN"
        else: new_turn_direction = "LEFT TURN" if is_wall_on_left else "RIGHT TURN"

    return new_state, new_course_type, new_turn_direction, new_front_block_for_turn, new_spike_detection_time, is_spike, direction

#アウトコースの処理（遅延処理）
def handle_state_delay(spike_detection_time):
    """
    [OUT_COURSE_DELAY] 状態の処理。
    一定時間直進したら旋回状態に移行する。
    戻り値: new_state (str)
    """
    if spike_detection_time is None:
        print("エラー: 遅延タイマーが未設定でOUT_COURSE_DELAYに移行しました。")
        return "WALL_FOLLOWING"
    
    elapsed = time.time() - spike_detection_time
    if elapsed >= POST_SPIKE_DELAY_S:
        print(f"遅延完了 ({POST_SPIKE_DELAY_S}秒)。旋回のため停止します。")
        drive_velocity(0, 0)
        return "TURNING_OUT_COURSE"
    else:
        drive_velocity(FORWARD_SPEED_PCT, 0) # 遅延中は直進
        return "OUT_COURSE_DELAY"

#インコースの旋回
def handle_state_turning_in(front_block_for_turn, turn_direction, cap, model, out, sw, sh, fw, fh, disp):
    """
    [TURNING_IN_COURSE] 状態の処理 (ブロッキング)。
    角度推定と旋回を実行し、完了したら次状態を返す。
    戻り値: new_state (str)
    """
    # --- (kiken.py L348-421 のロジックをそのまま移植) ---
    if front_block_for_turn is None:
        print("エラー: インコース目標が未設定でTURNING_IN_COURSEに移行しました。90度旋回します。")
        target_angle = 90.0
    else:
        TIME_PER_DEGREE = 0.102; TURN_SPEED_FOR_CALIBRATION = 5
        print("α+δの角度推定と時間制御による旋回を開始します。"); cv2.waitKey(1500)
        ret, stable_frame = cap.read(); disp = stable_frame.copy(); target_angle = 90.0
        if ret:
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
    
    print("インコース旋回完了。再アライメントへ移行します。")
    return "POST_TURN_ALIGN"

#アウトコースの旋回
def handle_state_turning_out(turn_direction, cap, model, out, sw, sh, fw, fh):
    """
    [TURNING_OUT_COURSE] 状態の処理 (ブロッキング)。
    固定旋回と壁探索を実行し、完了したら次状態を返す。
    戻り値: new_state (str)
    """
    # --- (kiken.py L423-488 のロジックをそのまま移植) ---
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
    STABLE_FRAME_THRESHOLD = 10 

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

        cv2.putText(search_frame, f"Searching... Stable Count: {stable_wall_counter}/{STABLE_FRAME_THRESHOLD}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Control', search_frame); out.write(search_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        if stable_wall_counter >= STABLE_FRAME_THRESHOLD:
            drive_velocity(0, 0)
            print(f"安定した壁を{STABLE_FRAME_THRESHOLD}フレーム連続で確認。探索を終了します。")
            found_wall = True
            break

    print("アウトコース探索完了。再アライメントへ移行します。")
    return "POST_TURN_ALIGN"

#旋回後の処理
def handle_state_post_turn_align(all_info, side_info):
    """
    [POST_TURN_ALIGN] 状態の処理。
    アライメント完了後、新しい基準距離を設定し、状態を遷移する。
    戻り値: (new_state, new_First_Dg, new_lower_thr, new_upper_thr, new_course_type, new_turn_direction, new_spike_detection_time)
    """
    status = perform_alignment_step(all_info, side_info)
    
    # アライメントが完了した場合
    if status == 'ALIGNED':
        print("再アライメント完了。壁追従を再開します。")
        
        # 新しい基準距離を設定
        new_First_Dg = None
        new_lower_thr = None
        new_upper_thr = None
        
        if side_info:
            new_First_Dg = min(side_info, key=lambda b: b[1])[1]
            new_lower_thr = new_First_Dg - 0.15
            new_upper_thr = new_First_Dg + 0.15
            print(f"新しい基準距離: {new_First_Dg:.2f}m")
        
        # 状態をリセットしてWALL_FOLLOWINGに戻す準備
        return ("WALL_FOLLOWING", 
                new_First_Dg, new_lower_thr, new_upper_thr, 
                "STRAIGHT", "NONE", None)

    # アライメント中の場合
    # 状態はPOST_TURN_ALIGNのまま、各種変数は変更しない (Noneを返す)
    return "POST_TURN_ALIGN", None, None, None, None, None, None

try:
    # --- 状態変数の初期化 ---
    current_state = "WAITING_TO_START"
    First_Dg  = None
    lower_thr = upper_thr = None
    spike_detection_time = None 
    
    # --- ループ内で更新・参照される変数 ---
    course_type = "STRAIGHT"
    turn_direction = "NONE"
    direction = 'straight'
    front_block_for_turn = None
    is_spike = False

    print("'s'キーで開始、'q'キーで終了します。")

    while True:
        # 1. 共通処理: 画像取得
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break
        disp = frame.copy()

        # 2. 共通処理: キー入力
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # 3. 共通処理: 画像処理と物体検出
        h, w = disp.shape[:2]
        scale_x, scale_y = w / sw, h / sh
        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.empty((0, 4))
        
        # 4. 共通処理: ブロック分類
        all_info, side_info, roi_info, front_info = classify_blocks(boxes, sw, sh, scale_x, scale_y)

        # 5. 状態に応じた処理の分岐
        #    各関数は新しい状態(new_state)を返す
        
        if current_state == "WAITING_TO_START":
            current_state = handle_state_waiting(key)

        elif current_state == "INITIAL_ALIGN":
            current_state = handle_state_initial_align(all_info, side_info)
            if current_state == "WALL_FOLLOWING":
                # アライメント完了時にのみFirst_Dgを初期設定
                if side_info:
                    First_Dg = min(b[1] for b in side_info)
                    print(f"最初の基準距離を設定: {First_Dg:.2f}m")
                    lower_thr = First_Dg - 0.15
                    upper_thr = First_Dg + 0.15
                else:
                    First_Dg = None # 稀だが、見失った場合はリセット

        elif current_state == "WALL_FOLLOWING":
            if First_Dg is None and side_info:
                # 旋回後などでFirst_Dgが未設定の場合、ここで再設定
                First_Dg = min(b[1] for b in side_info)
                print(f"基準距離を再設定: {First_Dg:.2f}m")
                lower_thr = First_Dg - 0.15
                upper_thr = First_Dg + 0.15

            (current_state, course_type, turn_direction, 
             front_block_for_turn, spike_detection_time, 
             is_spike, direction) = handle_state_wall_following(roi_info, side_info, First_Dg, lower_thr, upper_thr)

        elif current_state == "OUT_COURSE_DELAY":
            current_state = handle_state_delay(spike_detection_time)
            is_spike = True # 遅延中もスパイク表示を継続

        elif current_state == "TURNING_IN_COURSE":
            # この関数はブロッキング（完了するまで戻らない）
            current_state = handle_state_turning_in(
                front_block_for_turn, turn_direction, cap, model, out, sw, sh, fw, fh, disp
            )
            # 旋回が完了したら、次のループでPOST_TURN_ALIGNに入る

        elif current_state == "TURNING_OUT_COURSE":
            # この関数もブロッキング
            current_state = handle_state_turning_out(
                turn_direction, cap, model, out, sw, sh, fw, fh
            )
            # 旋回が完了したら、次のループでPOST_TURN_ALIGNに入る

        elif current_state == "POST_TURN_ALIGN":
            (new_state, new_First_Dg, new_lower_thr, new_upper_thr, 
             new_course_type, new_turn_direction, new_spike_time) = handle_state_post_turn_align(all_info, side_info)
            
            current_state = new_state
            if current_state == "WALL_FOLLOWING":
                # アライメント完了時、返された新しい値で状態変数を上書き
                First_Dg = new_First_Dg
                lower_thr = new_lower_thr
                upper_thr = new_upper_thr
                course_type = new_course_type
                turn_direction = new_turn_direction
                spike_detection_time = new_spike_time
                # ループ変数をリセット
                is_spike = False
                direction = 'straight'
                front_block_for_turn = None


        # 6. 共通処理: 描画
        draw_overlay(disp, current_state, course_type, direction, is_spike)
        
        # 7. 共通処理: 映像保存 & 表示
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