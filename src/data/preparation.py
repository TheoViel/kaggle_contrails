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


def prepare_data(data_path="../input/", processed_folder="false_color/"):
    df = pd.read_csv(os.path.join(data_path, processed_folder, "df.csv"))
    return df
