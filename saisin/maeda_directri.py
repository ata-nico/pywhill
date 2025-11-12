import cv2
import numpy as np
import os
import glob # ディレクトリ内のファイルを検索するために追加

def generate_custom_layout_with_rotation(equi_img_path, face_size, output_path):
    """
    1枚の正距円筒図法画像からカスタムレイアウトのキューブマップを生成し、指定されたパスに保存する関数。

    Args:
        equi_img_path (str): 入力画像のパス。
        face_size (int): 生成するキューブマップの各面の1辺のピクセル数。
        output_path (str): 出力画像の完全なファイルパス。
    """
    # 元画像読み込み
    img = cv2.imread(equi_img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"エラー: 画像ファイルを読み込めませんでした。パスを確認してください: {equi_img_path}")
        return

    h, w = img.shape[:2]
    to_rad = np.pi / 180.0

    # 各面の (名前, yaw, pitch)
    # posz: 前面(2), posx: 右面(5), negx: 左面(4), negy: 下面(3), negz: 背面(6)
    faces = {
        "posx": ( 90,   0),  # 右面 (5)
        "negx": (-90,   0),  # 左面 (4)
        "posy": (  0,  90),  # 上面 (1) ※未配置
        "negy": (  0, -90),  # 下面 (3)
        "posz": (  0,   0),  # 前面 (2)
        "negz": (180,   0),  # 背面 (6)
    }

    face_imgs = {}

    # フェイスごとに切り出し
    for name, (yaw_c, pitch_c) in faces.items():
        u = np.linspace(-1, 1, face_size)
        v = np.linspace(-1, 1, face_size)
        uu, vv = np.meshgrid(u, -v)
        x, y, z = uu, vv, -np.ones_like(uu)
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm; y /= norm; z /= norm

        # 回転行列
        pitch = pitch_c * to_rad
        yaw   = yaw_c   * to_rad
        Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
        R  = Ry @ Rx

        # 座標変換してマップ作成
        vec     = np.stack([x, y, z], axis=-1)
        vec_rot = vec @ R.T
        vx, vy, vz = vec_rot[...,0], vec_rot[...,1], vec_rot[...,2]
        lon = np.arctan2(vx, vz)
        lat = np.arcsin(vy)
        u_e = (lon/np.pi + 1)*0.5*(w-1)
        v_e = (1 - (lat/(0.5*np.pi) + 1)*0.5)*(h-1)
        map_x = u_e.astype(np.float32)
        map_y = v_e.astype(np.float32)

        face_imgs[name] = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    # キャンバス生成 (3行×3列)
    canvas = np.zeros((face_size * 3, face_size * 3, 3), dtype=np.uint8)

    # 新しい配置と回転指定
    # 5(posx), 4(negx), 6(negz) をそれぞれ 90CCW, 90CW, 180
    layout = {
        "posz": ((0, 1), None),                             # 前面 (2)
        "posx": ((1, 0), cv2.ROTATE_90_COUNTERCLOCKWISE),     # 右面 (5) -> 90° CCW
        "negy": ((1, 1), None),                             # 下面 (3)
        "negx": ((1, 2), cv2.ROTATE_90_CLOCKWISE),           # 左面 (4) -> 90° CW
        "negz": ((2, 1), cv2.ROTATE_180),                    # 背面 (6) -> 180°
    }

    # フェイスごとに回転＆貼り付け
    for name, ((r, c), rot_flag) in layout.items():
        if name not in face_imgs:
            continue
        face = face_imgs[name]
        if rot_flag is not None:
            face = cv2.rotate(face, rot_flag)
        y0, x0 = r * face_size, c * face_size
        canvas[y0:y0+face_size, x0:x0+face_size] = face

    # 全体を180°回転
    canvas = cv2.rotate(canvas, cv2.ROTATE_180)

    # 保存
    cv2.imwrite(output_path, canvas)
    print(f"✓ 変換完了: {output_path}")

if __name__ == "__main__":
    # --- 設定項目 ---
    # 処理したい画像が入っているディレクトリのパス
    input_dir = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img"
    # 変換後の画像を保存するディレクトリのパス
    output_dir  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\maeda\maeda_img"
    # 生成するキューブマップの1面のサイズ（ピクセル）
    face_sz  = 1024
    # ----------------

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 入力ディレクトリから指定した拡張子の画像ファイルを探す
    # (例: .jpg, .jpeg, .png に対応)
    supported_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"エラー: 指定されたディレクトリに画像が見つかりません。: {input_dir}")
    else:
        print(f"{len(image_paths)} 件の画像ファイルを処理します。")

        # 見つかった画像を1枚ずつ処理するループ
        for img_path in image_paths:
            print(f"変換中: {img_path}")
            
            # 出力ファイル名を生成 (例: input.jpg -> input_custom.jpg)
            base_name = os.path.basename(img_path)
            file_name, file_ext = os.path.splitext(base_name)
            output_filename = f"{file_name}_custom{file_ext}"
            output_path = os.path.join(output_dir, output_filename)

            # メインの変換関数を呼び出す
            generate_custom_layout_with_rotation(img_path, face_sz, output_path)

        print("\nすべての処理が完了しました。")