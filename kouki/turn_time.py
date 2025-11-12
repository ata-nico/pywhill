import time
import threading
# ご自身の環境に合わせて、whillライブラリのインポートを有効にしてください
from whill import ComWHILL

# --- パラメータ設定 ---
ROTATION_SPEED_PCT = 40  # 旋回速度 (%)
COMMAND_INTERVAL = 0.1   # 旋回命令を送信する間隔（秒）。0.1秒ごとに命令を再送信する

# --- WHILL 初期化 ---
whill = None
try:
    # 実機で動かす場合は、以下のコメントを解除してください
    whill = ComWHILL(port='COM3')
    whill.set_power(True)
    time.sleep(1)
    print("WHILL: 初期化処理をスキップ（シミュレーションモード）")
except Exception as e:
    print(f"WHILLへの接続に失敗しました: {e}")
    whill = None

# --- ヘルパー関数 ---
def clamp(x, a, b):
    return max(a, min(b, x))

def drive_velocity(f_pct, s_pct=0):
    f_pct = clamp(f_pct, -100, 100)
    s_pct = clamp(s_pct, -100, 100)
    if whill:
        try:
            whill.send_velocity(int(f_pct), int(s_pct / 100.0 * 1500))
        except Exception as e:
            print(f"WHILL 送信エラー: {e}")
    else:
        # print(f"[SIM] drive_velocity: F={f_pct}%, S={s_pct}%")
        pass

# --- 旋回命令を送り続けるための関数 ---
# この関数がバックグラウンドで動き続けます
def keep_turning(stop_event):
    print("バックグラウンドで旋回命令の送信を開始しました。")
    while not stop_event.is_set():
        drive_velocity(0, ROTATION_SPEED_PCT)
        time.sleep(COMMAND_INTERVAL)
    print("バックグラウンドの旋回命令送信を停止しました。")

# --- メインプログラム ---
try:
    print("\n" + "="*40)
    print("旋回タイム計測プログラム (連続回転対応版)")
    print(f"設定速度: {ROTATION_SPEED_PCT}%")
    print("="*40)

    # ユーザーの準備が整うのを待つ
    input("準備ができたら Enterキー を押して旋回を開始します...")
    
    # バックグラウンド処理を開始する合図を作成
    stop_event = threading.Event()
    turn_thread = threading.Thread(target=keep_turning, args=(stop_event,))
    
    # バックグラウンド処理を開始
    turn_thread.start()
    
    print("\n旋回を開始しました！ (ストップウォッチをスタート)")
    start_time = time.time()
    
    # ユーザーが停止の合図をするのを待つ
    input("ちょうど一周したら Enterキー を押して停止します...")
    
    end_time = time.time()
    
    # バックグラウンド処理に停止の合図を送る
    stop_event.set()
    turn_thread.join() # バックグラウンド処理が完全に終わるのを待つ
    
    print("\n旋回を停止しました。 (ストップウォッチをストップ)")
    drive_velocity(0, 0) # 完全に停止
    
    # 結果表示
    elapsed_time = end_time - start_time
    print("\n" + "="*30)
    print("  計測完了！")
    print(f"  速度 {ROTATION_SPEED_PCT} での一周(360°): {elapsed_time:.3f} 秒")
    print(f"  1度あたりの時間: {elapsed_time / 360.0:.4f} 秒")
    print("="*30)

finally:
    # 終了処理
    if whill:
        try:
            drive_velocity(0, 0)
            # whill.set_power(False)
        except Exception as e:
            print(f"WHILL 終了処理エラー: {e}")
    
    print("プログラムを終了します。")