import random
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class CropNonEmptyMaskIfExists(albu.CropNonEmptyMaskIfExists):
    """
    Modified class from:
    https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/crops/transforms.py#L163
    to allow for soft masks.
    """
    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)

        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))  # need copy as we perform in-place mod afterwards
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        mask_height, mask_width = mask.shape[:2]

        if (mask > 0.5).any():
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask > 0.5)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params


def blur_transforms(p=0.5, blur_limit=5):
    """
    Applies MotionBlur or GaussianBlur random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.
        blur_limit (int, optional): Blur intensity limit. Defaults to 5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(always_apply=True),
            albu.GaussianBlur(always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.
    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.RandomGamma(gamma_limit=(50, 150), always_apply=True),
            albu.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.2, always_apply=True
            ),
        ],
        p=p,
    )


def get_transfos(augment=True, resize=(256, 256), mean=0, std=1, strength=0, crop=False):
    """
    Returns transformations for image augmentation.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        resize (tuple, optional): Resize dimensions (height, width) of the images. Defaults to (256, 256).
        mean (np array, optional): Mean for normalization. Defaults to 0.
        std (np array, optional): Standard deviation for normalization. Defaults to 1.
        strength (int, optional): Strength level for augmentations. Defaults to 1.
        crop (bool, optional): Whether to apply CropNonEmptyMaskIfExists. Defaults to False.

    Returns:
        albumentation transforms: Transforms for image augmentation.
    """
    resize_aug = [
        CropNonEmptyMaskIfExists(resize[0], resize[1], always_apply=True),
    ] if crop else []

    normalizer = albu.Compose(
        resize_aug
        + [
            albu.Normalize(mean, std),
            ToTensorV2(),
        ],
        p=1,
    )

    if augment:
        if strength == 0:
            augs = []
        elif strength == 1:
            augs = [
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=30,
                    p=0.25,
                ),
            ]
        elif strength == 2:
            augs = [
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=30,
                    p=0.5,
                ),
                color_transforms(p=0.25),
                blur_transforms(p=0.1),
            ]
        elif strength == 3:
            augs = [
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=30,
                    p=0.75,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
            ]
    else:
        augs = []

    return albu.Compose(augs + [normalizer])
