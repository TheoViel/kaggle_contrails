import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import ContrailDataset
from data.transforms import get_transfos
from model_zoo.models import define_model
from util.torch import load_model_weights


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False, num_workers=8):
    """
    Perform model inference on a dataset.

    Args:
        model (nn.Module): Trained model for inference.
        dataset (Dataset): Dataset to perform inference on.
        loss_config (dict): Loss configuration.
        batch_size (int, optional): Batch size for inference. Defaults to 64.
        device (str, optional): Device to use for inference. Defaults to "cuda".
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.

    Returns:
        preds (numpy.ndarray): Predicted probabilities of shape (num_samples, num_classes).
        preds_aux (numpy.ndarray): Auxiliary predictions of shape (num_samples, num_aux_classes).
    """
    model.eval()
    preds, preds_aux = [], []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for img, _, _ in tqdm(loader):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred, _ = model(img.cuda())

            pred = pred[:, :1]  # 1st class only

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                pred = pred.softmax(-1)

#             if loss_config.get("activation_aux", "softmax") == "sigmoid":
#                 pred_aux = pred_aux.sigmoid()
#             elif loss_config.get("activation_aux", "softmax") == "softmax":
#                 pred_aux = pred_aux.softmax(-1)

            preds.append(pred.detach().cpu().numpy())
#             preds_aux.append(pred_aux.cpu().numpy())

    return np.concatenate(preds), []  #np.concatenate(preds_aux)


def kfold_inference(
    df,
    exp_folder,
    debug=False,
    use_fp16=False,
    save=False,
    num_workers=8,
    batch_size=None,
):
    """
    Perform k-fold cross-validation for model inference on the validation set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.

    Returns:
        np.ndarray: Array containing the predicted probabilities for each class.
    """
    config = Config(json.load(open(exp_folder + "config.json", "r")))

    model = define_model(
        config.decoder_name,
        config.encoder_name,
        num_classes=config.num_classes,
        n_channels=config.n_channels,
        reduce_stride=config.reduce_stride,
        use_pixel_shuffle=config.use_pixel_shuffle,
        use_hypercolumns=config.use_hypercolumns,
        center=config.center,
        use_cls=config.loss_config['aux_loss_weight'] > 0,
        pretrained=False,
    ).cuda()
    model = model.cuda().eval()

    preds = []
    for fold in config.selected_folds:
        print(f"\n- Fold {fold + 1}")

        weights = exp_folder + f"{config.decoder_name}_{config.encoder_name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)

        dataset = ContrailDataset(
            df, transforms=get_transfos(augment=False),
        )

        pred, _ = predict(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
            use_fp16=use_fp16,
            num_workers=num_workers,
        )
        
        if save:
            np.save(exp_folder + f"pred_val_{fold}.npy", pred)

        preds.append(pred)

    return np.mean(preds, 0)