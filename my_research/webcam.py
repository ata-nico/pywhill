import cv2
from ultralytics import YOLO
import sys

# YOLOv8のモデルをロード
model_path = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
try:
    model = YOLO(model_path)
    print(">>> INFO: YOLOモデルのロード成功", flush=True)
except Exception as e:
    print(f">>> ERROR: YOLOモデルのロード失敗: {e}", flush=True)
    exit()

# Webカメラの起動
camera_id = 0
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print(f">>> ERROR: カメラ {camera_id} を開けませんでした。", flush=True)
    exit()
print(f">>> INFO: カメラ {camera_id} を正常に開きました。", flush=True)

yolo_input_width = 640
yolo_input_height = 320
final_display_width_test = 200  # ウィンドウの幅を小さく

print(">>> INFO: メインループに入ります...", flush=True)
loop_count = 0
max_loops = 20  # テストのために制限

while cap.isOpened() and loop_count < max_loops:
    ret, frame = cap.read()
    if not ret:
        print(">>> ERROR: フレームをキャプチャできませんでした。", flush=True)
        break

    # YOLO用フレームをリサイズして推論
    try:
        yolo_frame = cv2.resize(frame, (yolo_input_width, yolo_input_height))
        results = model(yolo_frame)

        # YOLO推論結果の画像を取得（バウンディングボックス付き）
        result_img = results[0].plot()

        # 表示用サイズにリサイズ（ここがウィンドウを小さくする決定打）
        aspect_ratio = result_img.shape[0] / result_img.shape[1]
        final_display_height_test = int(final_display_width_test * aspect_ratio)
        resized_img = cv2.resize(result_img, (final_display_width_test, final_display_height_test))

        # ウィンドウを小さく指定（必要に応じて）
        cv2.namedWindow('YOLO Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Camera', final_display_width_test, final_display_height_test)

        # 表示
        print(f">>> TARGET PRINT: 表示テスト用フレームのシェイプ: {resized_img.shape}", flush=True)
        cv2.imshow('YOLO Camera', resized_img)

    except Exception as e:
        print(f">>> ERROR: 処理中にエラー: {e}", flush=True)
        import traceback
        traceback.print_exc()

    loop_count += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(">>> INFO: 'q'キーが押されました。ループを終了します。", flush=True)
        break

# 終了処理
print(f">>> INFO: メインループを終了しました (実行ループ数: {loop_count})。", flush=True)
cap.release()
cv2.destroyAllWindows()
print(">>> INFO: OpenCVウィンドウを全て破棄しました。プログラム終了。", flush=True)
