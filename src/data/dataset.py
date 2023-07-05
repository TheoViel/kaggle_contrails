import os
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ContrailDataset(Dataset):
    """
    Image torch Dataset.
    TODO

    Methods:
        __init__(df, transforms): Constructor
        __len__(): Get the length of the dataset
        __getitem__(idx): Get an item from the dataset

    Attributes:
        df (pandas DataFrame): Metadata
        img_paths (numpy array): Paths to the images
        mask_paths (numpy array): Paths to the masks
        transforms (albumentation transforms): Transforms to apply
        targets (numpy array): Target labels
    """

    def __init__(
        self,
        df,
        transforms=None,
        use_soft_mask=False,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.transforms = transforms
        self.use_soft_mask = use_soft_mask

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values
        self.targets = df["has_contrail"].values

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.img_paths[idx])
        
#         image_ = cv2.imread(re.sub("false_color/", "reg/", self.img_paths[idx]))
#         image[:, :, 0] = image_.mean(-1).astype(np.uint8)

        if self.use_soft_mask:
            mask_path = self.mask_paths[idx]
            folder = mask_path[:-4].rsplit('/', 3)[0] + "/train/" + mask_path[:-4].split('/')[-1]
            indiv_masks_path = folder + "/human_individual_masks.npy"
            if os.path.exists(indiv_masks_path):
                mask = np.load(indiv_masks_path).mean(-1).squeeze(-1)
            else:
                mask = cv2.imread(mask_path, 0)
        else:
            mask = cv2.imread(self.mask_paths[idx], 0)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        mask = mask.unsqueeze(0).float()  # H x W x 1

        y = torch.tensor([self.targets[idx]], dtype=torch.float)

        return image, mask, y
