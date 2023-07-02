import cv2
import torch
import numpy as np
import albumentations as albu

from albumentations import pytorch as AT
from torch.utils.data import Dataset

from util.boxes import Boxes


class InferenceDataset(Dataset):
    """
    Dataset for inference in a detection task.

    Attributes:
        df (DataFrame): The DataFrame containing the dataset information.
        paths (numpy.ndarray): The paths to the images in the dataset.
        transforms: Augmentations to apply to the images.
        pad (bool): Whether to apply padding to the images.
        pad_advanced (bool): Whether to apply advanced padding to the images.
        gts (list): Ground truth boxes for each image.
        classes (list): Ground truth classes for each image.

    Methods:
        __init__(self, df, transforms=None, pad=False, pad_advanced=False): Constructor
        __len__(self): Returns the length of the dataset.
        __getitem__(self, idx): Returns the item at the specified index.
    """

    def __init__(self, df, transforms=None, pad=False, pad_advanced=False):
        """
        Constructor

        Args:
            df (DataFrame): The DataFrame containing the dataset information.
            transforms (albu transforms, optional): Augmentations. Defaults to None.
            pad (bool, optional): Whether to apply padding to the images. Defaults to False.
            pad_advanced (bool, optional): Whether to apply advanced padding to the images. Defaults to False.
        """
        self.df = df
        self.paths = df["path"].values
        self.transforms = transforms
        self.pad = pad
        self.pad_advanced = pad_advanced

        self.gts, self.classes = [], []
        for i in range(len(df)):
            try:
                with open(df["gt_path"][i], "r") as f:
                    bboxes = np.array([line[:-1].split() for line in f.readlines()]).astype(float)
                    labels, bboxes = bboxes[:, 0], bboxes[:, 1:]
                    self.gts.append(bboxes)
                    self.classes.append(labels)
            except Exception:
                self.gts.append([])
                self.classes.append([])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Args:
            idx (int): Index.

        Returns:
            tuple: A tuple containing the image, ground truth, and image shape.
        """
        image = cv2.imread(self.paths[idx])

        if self.pad:
            if image.shape[1] > image.shape[0] * 1.4:
                padding = 255 * np.ones(
                    (int(image.shape[1] * 0.9) - image.shape[0], image.shape[1], image.shape[2]),
                    dtype=image.dtype,
                )
                image = np.concatenate([image, padding], 0)

            if self.pad_advanced:
                if image.shape[1] < image.shape[0] * 0.9:
                    padding = 255 * np.ones(
                        (image.shape[0], int(image.shape[0] * 1) - image.shape[1], image.shape[2]),
                        dtype=image.dtype,
                    )
                    image = np.concatenate([image, padding], 1)

        shape = image.shape

        if self.transforms is not None:
            try:
                image = self.transforms(image=image, bboxes=[], class_labels=[])["image"]
            except ValueError:
                image = self.transforms(image=image)["image"]

        return image, self.gts[idx], shape


def get_transfos(size):
    """
    Returns a composition of image transformations for preprocessing.

    Args:
        size (tuple): The desired size of the transformed image (height, width).

    Returns:
        albumentations.Compose: The composition of image transformations.
    """
    normalizer = albu.Compose(
        [
            albu.Normalize(mean=0, std=1),
            AT.transforms.ToTensorV2(),
        ],
        p=1,
    )

    return albu.Compose(
        [
            albu.Resize(size[0], size[1]),
            normalizer,
        ],
        bbox_params=albu.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


class DetectionMeter:
    """
    Detection meter for evaluating object detection performance.

    Methods:
        __init__(pred_format, truth_format): Constructor
        update(y_batch, preds, shape): Update ground truths and predictions
        reset(): Resets all values

    Attributes:
        truth_format (str): Format of ground truth bounding box coordinates
        pred_format (str): Format of predicted bounding box coordinates
        preds (list): List of predicted bounding boxes (Boxes instances)
        labels (list): List of labels corresponding to predicted bounding boxes
        confidences (list): List of confidence scores for predicted bounding boxes
        truths (list): List of ground truth bounding boxes (Boxes instances)
        metrics (dict): Dictionary storing evaluation metrics (tp, fp, fn, precision, recall, f1_score)
    """

    def __init__(self, pred_format="coco", truth_format="yolo"):
        """
        Constructor

        Args:
            pred_format (str, optional): Format of predicted bounding box coordinates. Defaults to "coco".
            truth_format (str, optional): Format of ground truth bounding box coordinates. Defaults to "yolo".
        """
        self.truth_format = truth_format
        self.pred_format = pred_format
        self.reset()

    def update(self, y_batch, preds, shape):
        """
        Update ground truths and predictions.

        Args:
            y_batch (list of np arrays): Truths.
            preds (list of torch tensors): Predictions.
            shape (list or tuple): Image shape.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        n, c, h, w = shape  # TODO : verif h & w

        self.truths += [
            Boxes(box, (h, w), bbox_format=self.truth_format) for box in y_batch
        ]

        for pred in preds:
            pred = pred.cpu().numpy()

            if pred.shape[1] >= 5:
                label = pred[:, 5].astype(int)
                self.labels.append(label)

            pred, confidences = pred[:, :4], pred[:, 4]

            self.preds.append(Boxes(pred, (h, w), bbox_format=self.pred_format))
            self.confidences.append(confidences)

    def reset(self):
        """
        Resets everything.
        """
        self.preds = []
        self.labels = []
        self.confidences = []
        self.truths = []

        self.metrics = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }


def collate_fn_val_yolo(batch):
    """
    Validation batch collating function for yolo-v5.

    Args:
        batch (tuple): Input batch.

    Returns:
        torch tensor: Images.
        list: Boxes.
        list: Image shapes.
    """
    img, boxes, shapes = zip(*batch)
    return torch.stack(list(img), 0), boxes, shapes


def nms(bounding_boxes, confidence_score, threshold=0.5):
    """
    Applies non-maximum suppression (NMS) to eliminate overlapping bounding boxes.

    Args:
        bounding_boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
        confidence_score (list): List of confidence scores corresponding to each bounding box.
        threshold (float, optional): Intersection-over-Union (IoU) threshold. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the picked bounding boxes and their corresponding scores.
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # Coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of the bounding box with the largest confidence score
        index = order[-1]

        # Pick the bounding box with the largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute coordinates of intersection-over-union (IoU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes), np.array(picked_score)
