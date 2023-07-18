import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def prepare_folds(data_path="../input/", k=4):
    """
    Prepare data folds for cross-validation.
    Original validation data is used as fold 0.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        k (int, optional): Number of cross-validation folds. Defaults to 4.

    Returns:
        pandas DataFrame: DataFrame containing the record IDs and their respective fold assignments.
    """
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
    """
    Prepare data for training or evaluation.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        processed_folder (str, optional): Name of the processed data folder. Defaults to "false_color/".
        use_raw (bool, optional): Whether to use raw images. Defaults to False.

    Returns:
        pandas DataFrame: DataFrame containing the prepared data for training or evaluation.
    """
    df = pd.read_csv(os.path.join(data_path, processed_folder, "df.csv"))
    if use_raw:
        df['img_path'] = df['folder']
    return df


def load_record(record_id, folder="../input/train/", load_mask=True, bands_to_load=(15, 14, 11)):
    """
    Load data for a specific record.

    Args:
        record_id (str): Identifier of the record to load.
        folder (str, optional): Path to the folder containing the data. Defaults to "../input/train/".
        load_mask (bool, optional): Whether to load masks. Defaults to True.
        bands_to_load (tuple, optional): Tuple containing the band numbers to load. Defaults to (15, 14, 11).

    Returns:
        tuple: A tuple containing two dictionaries: bands and masks.
            - bands (dict): Dictionary with band number as key and the corresponding band data as value.
            - masks (dict): Dictionary with mask name as key and the corresponding mask data as value.
                           This dictionary is empty if load_mask is False.
    """
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
    """
    Maps data to the range [0, 1].

    Args:
        data (numpy array): Input data to be normalized.
        bounds (tuple, optional): A tuple containing the lower and upper bounds for normalization.
            If not provided, the minimum and maximum values of the input data will be used.

    Returns:
        numpy array: The normalized data with values in the range [0, 1].
    """
    if bounds is None:
        bounds = (np.min(data), np.max(data))
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_false_color_img(bands):
    """
    Generates a false-color image from input bands.

    Args:
        bands (dict): A dictionary containing the bands to be used for generating the false-color image.

    Returns:
        numpy array: A false-color image represented as a numpy array with values in the range [0, 1].
    """
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)

    r = normalize_range(bands[15] - bands[14], _TDIFF_BOUNDS)
    g = normalize_range(bands[14] - bands[11], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(bands[14], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    return false_color
