import cv2
from ultralytics import YOLO # ultralyticsライブラリをインポート

# YOLOv8のモデルをロード (以前指定されたパスを使用)
try:
    model_path = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
    #model_path = "yolov8s.pt"
    model = YOLO(model_path)
    print(f">>> INFO: YOLOモデルのロード成功: {model_path}", flush=True)
except Exception as e:
    print(f">>> ERROR: YOLOモデルのロードに失敗しました: {e}", flush=True)
    exit()

def show_yolo_detection():
    # カメラID（環境に合わせて調整してください）
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"エラー: カメラ {camera_id} を開けませんでした。", flush=True)
        return

    print("カメラを起動し、YOLOリアルタイム検出を開始します。ウィンドウを閉じるか 'q' キーを押すと終了します。", flush=True)

    # YOLOモデルに入力する画像の目標サイズ
    yolo_input_width = 640
    yolo_input_height = 320 # YOLOログで (1, 3, 320, 640) だったので、高さを320に

    # 最終的に画面に表示するウィンドウの目標幅（ピクセル単位）
    # この値を調整して、表示ウィンドウの大きさを変更できます (例: 640, 800, 960など)
    final_display_width = 800 # ノートパソコンで程よいサイズとして一旦800に設定（お好みで変更）

    loop_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームをキャプチャできませんでした。", flush=True)
            break
        
        # print(f"ループ {loop_count}: 元フレームサイズ {frame.shape}", flush=True) # デバッグ用

        # 1. YOLO処理用のフレームをリサイズ
        try:
            yolo_processing_frame = cv2.resize(frame, (yolo_input_width, yolo_input_height))
        except Exception as e:
            print(f"エラー: YOLO用フレームのリサイズ中にエラー: {e}", flush=True)
            continue # 次のフレームへ

            # 2. YOLOで物体検出
        try:
            results = model(yolo_processing_frame, verbose=False)
            result = results[0]

            # 信頼度0.7以上の検出のみを残す
            confidence_threshold = 0.6
            boxes = result.boxes
            if boxes is not None:
                mask = boxes.conf > confidence_threshold
                boxes = boxes[mask]
                result.boxes = boxes

        except Exception as e:
            print(f"エラー: YOLO推論中にエラー: {e}", flush=True)
            continue

        # 3. 結果をフレームに描画 (yolo_processing_frame と同じサイズ)
        try:
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"エラー: 検出結果の描画中にエラー: {e}", flush=True)
            annotated_frame = yolo_processing_frame # エラー時は処理前フレームをとりあえず使う

        # 4. 表示用に annotated_frame をさらにリサイズ
        try:
            ann_height, ann_width = annotated_frame.shape[:2]
            if ann_width == 0: # 幅が0の場合のエラー回避
                display_frame = annotated_frame # そのまま使う
            else:
                aspect_ratio_ann = ann_height / ann_width
                final_display_height = int(final_display_width * aspect_ratio_ann)
                if final_display_width <= 0 or final_display_height <= 0: # 無効なサイズチェック
                    display_frame = annotated_frame # そのまま使う
                else:
                    display_frame = cv2.resize(annotated_frame, (final_display_width, final_display_height))
        except Exception as e:
            print(f"エラー: 表示用フレームのリサイズ中にエラー: {e}", flush=True)
            display_frame = annotated_frame # エラー時はリサイズ前フレーム

        # 5. 最終的にリサイズされたフレームを表示
        cv2.imshow('YOLO Real-time Detection', display_frame)

        loop_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q'キーが押されました。終了します。", flush=True)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("カメラとウィンドウを解放し、終了しました。", flush=True)

if __name__ == '__main__':
    show_yolo_detection()