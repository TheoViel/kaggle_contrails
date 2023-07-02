import os
import gc
import sys
import torch
import importlib
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("../yolox")
sys.path.append("yolox")
from yolox.utils import postprocess  # noqa
from inference.utils import DetectionMeter, collate_fn_val_yolo  # noqa


def predict(model, dataset, config, disable_tqdm=True, extract_fts=False):
    """
    Performs prediction on a dataset using a model.

    Args:
        model (nn.Module): The model to use for prediction.
        dataset (Dataset): The dataset to perform prediction on.
        config: Configuration object or dictionary.
        disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to True.
        extract_fts (bool, optional): Whether to extract features from the model. Defaults to False.

    Returns:
        tuple: A tuple containing the evaluation meter and the extracted features (if extract_fts=True).
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=config.val_bs,
        shuffle=False,
        collate_fn=collate_fn_val_yolo,
        num_workers=2,
        pin_memory=True,
    )

    meter = DetectionMeter(pred_format=config.pred_format, truth_format=config.bbox_format)
    meter.reset()

    fts_list, fts = [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=disable_tqdm):
            x = batch[0].to(config.device)

            try:
                pred_boxes, fts = model(x)
            except Exception:
                pred_boxes = model(x)

            meter.update(batch[1], pred_boxes, x.size())

            if extract_fts:
                fts_list += fts

    gc.collect()
    torch.cuda.empty_cache()

    if extract_fts:
        return meter, fts_list
    else:
        return meter


def retrieve_yolox_model(exp_file, ckpt_file, size=(1024, 1024), verbose=1):
    """
    Retrieves and configures a YOLOX model for inference.

    Args:
        exp_file (str): The path to the experiment file.
        ckpt_file (str): The path to the checkpoint file containing the model weights.
        size (tuple, optional): The input size of the model. Defaults to (1024, 1024).
        verbose (int, optional): Verbosity level. If 1, it prints the loading message. Defaults to 1.

    Returns:
        nn.Module: The configured YOLOX model for inference.
    """
    sys.path.append(os.path.dirname(exp_file))
    current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])

    exp = current_exp.Exp()

    exp.test_conf = 0.0
    exp.test_size = size
    exp.nmsthre = 0.75

    model_roi_ = exp.get_model()

    if verbose:
        print(" -> Loading weights from", ckpt_file)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model_roi_.load_state_dict(ckpt["model"], strict=True)

    model_roi_.max_det = 100
    model_roi_.nmsthre = 0.75
    model_roi_.test_conf = 0.1
    model_roi_.test_size = exp.test_size
    model_roi_.num_classes = 1
    model_roi_.stride = 64
    model_roi_.amp = False  # FP16

    return model_roi_.eval().cuda()


class YoloXWrapper(nn.Module):
    """
    Wrapper for YoloX models.

    Methods:
        __init__(model, config): Constructor
        forward(x): Forward function

    Attributes:
        model (torch model): Yolo-v5 model.
        config (Config): Config.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IoU threshold.
        max_per_img (int): Maximum number of detections per image.
        min_per_img (int): Minimum number of detections per image.
    """
    def __init__(self, model, config):
        """
        Constructor

        Args:
            model (torch model): Yolo model.
            config (Config): Config.
        """
        super().__init__()
        self.model = model
        self.config = config

        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh
        self.max_per_img = config.max_per_img
        self.min_per_img = config.min_per_img

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Input images.

        Returns:
            torch tensor: Predictions.
        """
        pred_boxes, features = self.model(x * 255, return_fts=True)

        pred_boxes_ = None
        conf_thresh = self.conf_thresh
        while pred_boxes_ is None:
            pred_boxes_ = postprocess(
                pred_boxes.clone(), 1, conf_thresh, self.iou_thresh, class_agnostic=False,
            )[0]
            conf_thresh = max(conf_thresh - 0.1, conf_thresh / 10)

        return [pred_boxes_], [features]
