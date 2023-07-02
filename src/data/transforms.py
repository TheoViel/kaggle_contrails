import albumentations as albu
from albumentations.pytorch import ToTensorV2


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


def get_transfos(augment=True, resize=(256, 256), mean=0, std=1, strength=0):
    """
    Returns transformations for image augmentation.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        resize (tuple, optional): Resize dimensions (height, width) of the images. Defaults to (256, 256).
        mean (np array, optional): Mean for normalization. Defaults to 0.
        std (np array, optional): Standard deviation for normalization. Defaults to 1.
        strength (int, optional): Strength level for augmentations. Defaults to 1.

    Returns:
        albumentation transforms: Transforms for image augmentation.
    """
    resize_aug = []
    
#         albu.Resize(resize[0], resize[1])
#     ] if resize else []

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
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ]
        elif strength == 2:
            augs = [
#                 albu.HorizontalFlip(p=0.5),
#                 albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.2,
                    shift_limit=0.2,
                    rotate_limit=30,
                    p=0.5,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
            ]
        elif strength == 3:
            augs = [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.25,
                    shift_limit=0.2,
                    rotate_limit=45,
                    p=0.75,
                ),
                color_transforms(p=0.5),
                blur_transforms(p=0.25),
            ]
    else:
        augs = []

    return albu.Compose(augs + [normalizer])
