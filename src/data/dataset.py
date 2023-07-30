import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.transforms import get_transfos
from data.preparation import load_record, get_false_color_img


class ContrailDataset(Dataset):
    """
    Custom dataset for contrail data.

    Methods:
        __init__(df, transforms, use_soft_mask, use_shape_descript, use_pl_masks, frames):
            Constructor for the dataset.
        __len__(): Get the length of the dataset.
        __getitem__(idx): Get an item from the dataset.

    Attributes:
        df (pandas DataFrame): Metadata containing information about the dataset.
        transforms (albumentation transforms): Transforms to apply to the images and masks.
        use_soft_mask (bool): Flag indicating whether to use the soft mask or not.
        use_shape_descript (bool): Flag indicating whether to use shape descriptors.
        use_pl_masks (bool): Flag indicating whether to use pseudo-label masks.
        frames (int or list): Frame(s) to use for the false-color image.
        img_paths (numpy array): Array of paths to the images.
        mask_paths (numpy array): Array of paths to the masks.
        folders (numpy array): Array of folder paths for the images.
        ids (numpy array): Array of record IDs for the images.
        targets (numpy array): Array of target labels indicating the presence of contrails.
    """

    def __init__(
        self,
        df,
        transforms=None,
        use_soft_mask=False,
        use_shape_descript=False,
        use_pl_masks=False,
        frames=4,
        use_ext_data=False,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata containing information about the dataset.
            transforms (albumentation transforms, optional): Transforms to apply to the images and masks. Defaults to None.
            use_soft_mask (bool, optional): Flag indicating whether to use the soft mask or not. Defaults to False.
            use_shape_descript (bool, optional): Flag indicating whether to use shape descriptors. Defaults to False.
            use_pl_masks (bool, optional): Flag indicating whether to use pseudo-label masks. Defaults to False.
            frames (int or list, optional): Frame(s) to use for the false-color image. Defaults to 4.
        """
        self.df = df
        self.transforms = transforms
        self.use_soft_mask = use_soft_mask
        self.use_shape_descript = use_shape_descript
        self.use_pl_masks = use_pl_masks
        self.use_ext_data = use_ext_data
        self.frames = frames

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values
        self.folders = df["folder"].values
        self.ids = df["record_id"].values

        self.targets = df["has_contrail"].values
        
        self.ext_data_prop = 0
        if self.use_ext_data:
            self.df_ext = pd.read_csv('../output/df_goes16_may.csv')
            self.transfos_ext = get_transfos(strength=3, crop=True)
            self.ext_data_prop = 0.5

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)
    
    def _getitem_ext(self):
        idx = np.random.randint(len(self.df_ext))

        img_path = self.df_ext['img_path'].values[idx]
        mask_path = self.df_ext['mask_path'].values[idx]
        
        image = cv2.imread(img_path)
        mask = np.load(mask_path).astype(np.float32)

        transformed = self.transfos_ext(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()
        
        if self.use_shape_descript:
            from data.shape_descript import get_shape_descript
            shape_descript = get_shape_descript(
                segmentation=(mask[0].numpy() > 0.5), sigma=(10, 10), voxel_size=(1, 1), downsample=2
            )
            shape_descript[-1] *= 4
            shape_descript = np.clip(shape_descript, 0, 1)
            mask = torch.cat([mask, torch.from_numpy(shape_descript)], 0)
            
        y = (mask.max() > 0.5).float().view(1)
            
        return image, mask, y

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [C, H, W].
            torch.Tensor: Mask as a tensor of shape [1 or 7, H, W].
            torch.Tensor: Label as a tensor of shape [1].
        """
        if np.random.random() < self.ext_data_prop:
            return self._getitem_ext()
        
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
            if self.mask_paths[idx].endswith('.npy'):
                mask = np.load(self.mask_paths[idx]).astype(np.float32)
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

#         if image.size(1) > 256:  # Inf mode, this should be commented
#             d = 16
#             p1 = (d - image.size(1) % d) % d
#             p2 = (d - image.size(2) % d) % d
#             if p1 or p2:
#                 image = torch.nn.functional.pad(image, (0, p1, 0, p2, 0, 0))

        return image, mask, y


class ContrailInfDataset(Dataset):
    """
    Image torch Dataset for inference.

    Methods:
        __init__(folders, transforms): Constructor
        __len__(): Get the length of the dataset
        __getitem__(idx): Get an item from the dataset

    Attributes:
        folders (list): List of paths to the folders containing data for inference
        transforms (albumentation transforms): Transforms to apply
        frames (int or list or tuple): Frame indices or indices range to extract from false color images
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
            folders (list): List of paths to the folders containing data for inference
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
            frames (int or list or tuple, optional): Frame indices or indices range to extract from false color images. Defaults to 4.
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
            np array [H x W]: Mask or 0 if mask is not available.
            int: Sample weight (always 0).
        """
        bands, masks = load_record(self.folders[idx], folder="")
        false_color = get_false_color_img(bands)

        if isinstance(self.frames, int):
            image = false_color[..., self.frames]
        else:  # list, tuple
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
            
        if isinstance(image, torch.Tensor):
            if image.size(0) > 3:
                image = image.view(3, -1, image.size(1), image.size(2)).transpose(0, 1)

        return image, mask, 0
