import os
import shutil

# --- 設定 ---
SRC_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\label"
DST_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\raw_only"

# コピー先ディレクトリ作成
os.makedirs(DST_DIR, exist_ok=True)

# src_dir 内を走査して、"_raw.png" がついているファイルをコピー
for fname in os.listdir(SRC_DIR):
    if fname.endswith("_raw.png"):
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        shutil.copy2(src_path, dst_path)
        #移動したい場合は copy2 → os.rename か shutil.move を使ってください
        #shutil.move(src_path, dst_path)

print(f"Copied raw images to: {DST_DIR}")
