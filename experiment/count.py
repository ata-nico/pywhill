import os
from collections import Counter

# --- 設定: カウントしたいフォルダパス ---
SRC_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\raw_only"

# --- ファイル一覧取得 ---
files = os.listdir(SRC_DIR)

# --- ラベル抽出 ---
labels = []
for f in files:
    # raw画像だけ数えたい場合：
    if f.endswith("_raw.png"):
        # ファイル名: frame_00000_straight_raw.png
        # split("_") → ["frame","00000","straight","raw.png"]
        parts = f.split("_")
        if len(parts) >= 4:
            labels.append(parts[2])
    # annotated画像もまとめて数えたい場合は上記をコメントアウトし、下記を有効に
    # if f.endswith("_annotated.png"):
    #     parts = f.split("_")
    #     labels.append(parts[2])

# --- カウント ---
cnt = Counter(labels)

# --- 結果表示 ---
for label in ["straight", "turn", "no"]:
    print(f"{label:12s}: {cnt.get(label,0)} 枚")

# 例）
# straight     : 120 枚
# turn         : 80 枚
# no_detection : 10 枚
