import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_contours(img, mask=None, preds=None, w=1):
    """
    Plots the contours of mask predictions (in green) and of a mask (in red).

    Args:
        img (numpy array [H x W x C]): Image.
        preds (numpy int array [H x W] or None): Predicted mask.
        mask (numpy array [H x W] or None): Mask.
        w (int, optional): Contour width. Defaults to 1.
        downsize (int, optional): Downsizing factor. Defaults to 1.

    Returns:
        px.imshow: Ploty plot.
    """
    contours, contours_preds = None, None
    img = img.copy()
    if img.max() > 1:
        img = (img / 255).astype(float)

    if mask is not None:
        if mask.max() > 1:
            mask = (mask / 255).astype(float)
        mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if preds is not None:
        if preds.max() > 1:
            preds = (preds / 255).astype(float)
        preds = (preds * 255).astype(np.uint8)
        contours_preds, _ = cv2.findContours(preds, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
    if contours_preds is not None:
        cv2.polylines(img, contours_preds, True, (0.0, 1.0, 0.0), w)
    if contours is not None:
        cv2.polylines(img, contours, True, (1.0, 0.0, 0.0), w)
        
#  img = (img + img_gt) / 2

    plt.imshow(img)
    plt.axis(False)
    
    
def plot_mask(img, mask):
    mask = mask.copy()
    mask = np.where(mask, mask, img)

    plt.imshow(img)
    plt.imshow(mask, cmap='Reds', alpha=.4, interpolation='none')
    plt.axis(False)
    
    
def plot_sample(img, mask, figsize=(18, 6)):
    plt.figure(figsize=figsize)

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis(False)

    plt.subplot(1, 3, 2)
    plot_mask(img, mask)

    plt.subplot(1, 3, 3)
    plot_contours(img, mask)
    plt.show()
