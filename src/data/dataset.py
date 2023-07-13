import os
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from data.preparation import load_record, get_false_color_img


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
        use_shape_descript=False,
        use_pl_masks=False,
        frames=4,
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
        self.use_shape_descript = use_shape_descript
        self.use_pl_masks = use_pl_masks
        self.frames = frames

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values
        self.folders = df["folder"].values
        self.ids = df["record_id"].values

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
        path = self.img_paths[idx]

        if path.endswith('.png'):
            assert self.frames == 4
            image = cv2.imread(path)
        else:
            bands, _ = load_record(path, folder="", load_mask=False)
            false_color = get_false_color_img(bands)
            
            if isinstance(self.frames, int):
                image = false_color[..., self.frames]
            else: # list, tuple
                image = false_color[..., np.array(self.frames)]
                image = image.reshape(image.shape[0], image.shape[1], -1)

            image = (image * 255).astype(np.uint8)

        if self.use_soft_mask:
            indiv_masks_path = self.folders[idx] + "/human_individual_masks.npy"
            if os.path.exists(indiv_masks_path):
                mask = np.load(indiv_masks_path).mean(-1).squeeze(-1)
            else:
                mask = cv2.imread(self.mask_paths[idx], 0)
                
            if self.use_pl_masks:
                mask_pl = np.load(f'../logs/2023-07-06/23/pl_masks/{self.ids[idx]}.npy')
                mask = (mask + mask_pl.astype(np.float32)) / 2
        else:
            mask = cv2.imread(self.mask_paths[idx], 0)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
        mask = mask.unsqueeze(0).float()  # H x W x 1
        
        if self.use_shape_descript:
            from data.shape_descript import get_shape_descript
            shape_descript = get_shape_descript(
                segmentation=(mask[0].numpy() > 0.5), sigma=(10, 10), voxel_size=(1, 1), downsample=2
            )
            shape_descript[-1] *= 4
            shape_descript = np.clip(shape_descript, 0, 1)
            mask = torch.cat([mask, torch.from_numpy(shape_descript)], 0)

        y = torch.tensor([self.targets[idx]], dtype=torch.float)
        
        if image.size(0) > 3:
            image = image.view(3, -1, image.size(1), image.size(2)).transpose(0, 1)
            
#         print(image.shape)

        return image, mask, y


class ContrailInfDataset(Dataset):
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
        folders,
        transforms=None,
        frames=4,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.folders = folders
        self.transforms = transforms
        self.frames = frames

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.folders)

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
        bands, masks = load_record(self.folders[idx], folder="")
        false_color = get_false_color_img(bands)

        if isinstance(self.frames, int):
            image = false_color[..., self.frames]
        else: # list, tuple
            image = false_color[..., np.array(self.frames)]
            image = image.reshape(image.shape[0], image.shape[1], -1)
                
        image = (image * 255).astype(np.uint8)

        try:
            mask = masks['human_pixel_masks']
        except KeyError:
            mask = 0

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, mask, 0
