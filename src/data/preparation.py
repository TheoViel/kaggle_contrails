import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def prepare_folds(data_path="../input/", k=4):
    folders = glob.glob(data_path + "train/*") + glob.glob(data_path + "validation/*")
    
    kf = KFold(n_splits=k - 1, shuffle=True, random_state=333)
    
    splits = kf.split(folders)
    
    df_folds = pd.DataFrame({"record_id": [f.split('/')[-1] for f in folders], "folder": folders})
    
    df_folds['fold'] = -1
    for i, (_, val_idx) in enumerate(splits):
        df_folds.loc[val_idx, "fold"] = i
        
    df_folds['fold'] += 1
    df_folds.loc[  # validation is for 0
        df_folds[df_folds['folder'].apply(lambda x: "validation/" in x)].index, "fold"
    ] = 0
        
    df_folds[["record_id", "fold"]].to_csv(f"{data_path}/folds_{k}.csv", index=False)
    
    return df_folds[["record_id", "fold"]]


def prepare_folds(data_path="../input/", k=4):
    folders = glob.glob(data_path + "train/*") + glob.glob(data_path + "validation/*")
    
    kf = KFold(n_splits=k - 1, shuffle=True, random_state=333)
    
    splits = kf.split(folders)
    
    df_folds = pd.DataFrame({"record_id": [f.split('/')[-1] for f in folders], "folder": folders})
    
    df_folds['fold'] = -1
    for i, (_, val_idx) in enumerate(splits):
        df_folds.loc[val_idx, "fold"] = i
        
    df_folds['fold'] += 1
    df_folds.loc[  # validation is for 0
        df_folds[df_folds['folder'].apply(lambda x: "validation/" in x)].index, "fold"
    ] = 0
        
    df_folds[["record_id", "fold"]].to_csv(f"{data_path}/folds_{k}.csv", index=False)
    
    return df_folds[["record_id", "fold"]]


def prepare_data(data_path="../input/", processed_folder="false_color/", use_raw=False):
    df = pd.read_csv(os.path.join(data_path, processed_folder, "df.csv"))
    if use_raw:
        df['img_path'] = df['folder']
    return df


def load_record(record_id, folder="../input/train/", load_mask=True, bands_to_load=(15, 14, 11)):
    files = sorted(os.listdir(folder + record_id))

    bands, masks = {}, {}

    for f in files:
        if "band" in f:
            num = int(f.split(".")[0].split("_")[-1])
            if num in bands_to_load:
                bands[num] = np.load(os.path.join(folder, record_id, f))
        else:
            if load_mask:
                masks[f[:-4]] = np.load(os.path.join(folder, record_id, f))

    return bands, masks


def normalize_range(data, bounds=None):
    """Maps data to the range [0, 1]."""
    if bounds is None:
        bounds = (np.min(data), np.max(data))
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_false_color_img(bands):
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    r = normalize_range(bands[15] - bands[14], _TDIFF_BOUNDS)
    g = normalize_range(bands[14] - bands[11], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(bands[14], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    return false_color
