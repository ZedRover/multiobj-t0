import os
import numpy as np
from glob import glob
from tqdm import tqdm
import SharedArray as sa
import sutils
import common as cm


read_dir = "/mnt/disk2/factors_jun"
output_dir = "/mnt/disk2/factor_0626"
fold_all = [0, 1, 2, 3, 4, 5]


def get_label_data(code):
    load_dir = f"/mnt/disk2/factors_jun/{code}"
    tags = ["mean", "var", "vol", "min", "max", "gap"]
    horizons = ["0-60", "60-120", "60-300", "60-600", "120-180"]
    load_dir = f"/mnt/disk2/factors_jun/{code}"
    ts = sa.attach(f"timestamp_{code}")
    pad_len = sum(ts < 20210401)
    label_list = [
        np.vstack([np.load(f"{load_dir}/{i}/{tag}_{horizon}.npy") for i in range(6)])
        for tag in tags
        for horizon in horizons
    ]
    label_name = [f"{tag}_{horizon}" for tag in tags for horizon in horizons]
    label = np.hstack(label_list)
    label = np.pad(label, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0)

    return label, label_name


for code in tqdm(cm.SELECTED_CODES):
    label, label_name = get_label_data(code)
    save_dir = f"{output_dir}/{code}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/{code}.npy", label)

with open(f"{output_dir}/README.md", "w") as f:
    # save label name
    f.write("\n".join(label_name))
