import os

# --- 設定 ---
SRC_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\raw_only"

# 生画像だけ抽出してソート
raw_files = sorted([
    f for f in os.listdir(SRC_DIR)
    if f.endswith("_raw.png")
])

# 連番でリネーム
for new_idx, old_name in enumerate(raw_files):
    # 元のファイル名例: frame_000024_straight_raw.png
    parts = old_name.split("_")
    # parts = ["frame", "{oldnum}", "{direction}", "raw.png"]
    direction = parts[2]
    new_name = f"frame_{new_idx:05d}_{direction}_raw.png"
    os.rename(
        os.path.join(SRC_DIR, old_name),
        os.path.join(SRC_DIR, new_name)
    )

print(f"Renamed {len(raw_files)} raw images to sequential numbering.")
