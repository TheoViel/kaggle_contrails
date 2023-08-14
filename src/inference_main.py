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


def predict_multi(models, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False, num_workers=8):
    """
    Perform multi-model inference and generate predictions for the given dataset.

    Args:
        models (list of torch.nn.Module): List of trained models for inference.
        dataset (torch.utils.data.Dataset): Dataset for which to generate predictions.
        loss_config (dict): Configuration for loss function and activation.
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        device (str, optional): Device for inference, 'cuda' or 'cpu'. Defaults to 'cuda'.
        use_fp16 (bool, optional): Whether to use mixed-precision (FP16) inference. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.

    Returns:
        np array [N x C]: Predicted probabilities for each class for each sample.
        list: Empty list, placeholder for the auxiliary task.
    """
    for model in models:
        model.eval()
    preds = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for img, _, _ in tqdm(loader):
            preds_model = []
            for model in models:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    pred, _ = model(img.cuda())

                pred = pred[:, :1]  # 1st class only

                # Get probabilities
                if loss_config["activation"] == "sigmoid":
                    pred = pred.sigmoid()
                elif loss_config["activation"] == "softmax":
                    pred = pred.softmax(-1)

                preds_model.append(pred.detach().cpu().numpy())
            preds.append(np.mean(preds_model, 0))

    return np.concatenate(preds), []


def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False, num_workers=8):
    """
    Perform inference using a single model and generate predictions for the given dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataset (torch.utils.data.Dataset): Dataset for which to generate predictions.
        loss_config (dict): Configuration for loss function and activation.
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        device (str, optional): Device for inference, 'cuda' or 'cpu'. Defaults to 'cuda'.
        use_fp16 (bool, optional): Whether to use mixed-precision (FP16) inference. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.

    Returns:
        np array [N x C]: Predicted probabilities for each class for each sample.
        list: Empty list, placeholder for the auxiliary task.
    """
    model.eval()
    preds = []

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

            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds), []


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
        save (bool, optional): Whether to save the predictions. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.
        batch_size (int, optional): Batch size. If None, uses the value in the config. Defaults to None.

    Returns:
        List[np.ndarray]: List of arrays containing the predicted probabilities for each class for each fold.
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
        frames=config.frames if hasattr(config, "use_lstm") else 4,
        use_lstm=config.use_lstm if hasattr(config, "use_lstm") else False,
        bidirectional=config.bidirectional if hasattr(config, "bidirectional") else False,
        use_cnn=config.use_cnn if hasattr(config, "use_cnn") else False,
        kernel_size=config.kernel_size if hasattr(config, "kernel_size") else 1,
        use_transfo=config.use_transfo if hasattr(config, "use_transfo") else False,
    )
    model = model.cuda().eval()

    preds = []
    for fold in config.selected_folds:
        print(f"\n- Fold {fold + 1}")

        weights = exp_folder + f"{config.decoder_name}_{config.encoder_name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)

        df_val = df[df['fold'] == fold].reset_index(drop=True) if "fold" in df.columns else df

        dataset = ContrailDataset(
            df_val,
            transforms=get_transfos(augment=False),
            frames=config.frames if hasattr(config, "frames") else 4,
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

    return preds
