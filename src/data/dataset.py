import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ContrailDataset(Dataset):
    """
    Image torch Dataset.

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
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values
        self.transforms = transforms
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
        mask = cv2.imread(self.mask_paths[idx], 0)
        
#         plot_sample(image, mask[..., None])

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        mask = mask.unsqueeze(0).float()  # H x W x 1

        y = torch.tensor([self.targets[idx]], dtype=torch.float)

        return image, mask, y
