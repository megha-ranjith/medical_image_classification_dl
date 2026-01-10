import os
import shutil
import kagglehub

NORMAL_DIR = "data/raw/NORMAL"
TB_DIR = "data/raw/TB"

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

def copy_images(dataset_path):
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith(".png"):
                continue

            src = os.path.join(root, file)

            # STRICT filename-based labeling
            if file.endswith("_0.png"):
                shutil.copy(src, os.path.join(NORMAL_DIR, file))
            elif file.endswith("_1.png"):
                shutil.copy(src, os.path.join(TB_DIR, file))

# Download datasets
mc_path = kagglehub.dataset_download(
    "raddar/tuberculosis-chest-xrays-montgomery"
)
sz_path = kagglehub.dataset_download(
    "raddar/tuberculosis-chest-xrays-shenzhen"
)

copy_images(mc_path)
copy_images(sz_path)

print("✅ Dataset cleaned and copied correctly")
