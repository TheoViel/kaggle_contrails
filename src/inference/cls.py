import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from params import NUM_WORKERS
from data.dataset import ClsDataset
from data.transforms import get_transfos
from model_zoo.models import define_model
from util.torch import load_model_weights
from util.metrics import accuracy


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False):
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
    preds = np.empty((0, model.num_classes))
    preds_aux = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for img, _, _ in tqdm(loader):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred, pred_aux = model(img.cuda())

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                pred = pred.softmax(-1)

            if loss_config.get("activation_aux", "softmax") == "sigmoid":
                pred_aux = pred_aux.sigmoid()
            elif loss_config.get("activation_aux", "softmax") == "softmax":
                pred_aux = pred_aux.softmax(-1)

            preds = np.concatenate([preds, pred.cpu().numpy()])
            preds_aux.append(pred_aux.cpu().numpy())

    return preds, np.concatenate(preds_aux)


def kfold_inference(
    df,
    exp_folder,
    debug=False,
    use_tta=False,
    use_fp16=False,
):
    """
    Perform k-fold cross-validation for model inference on the validation set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        use_tta (bool, optional): Whether to use test time augmentation. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.

    Returns:
        np.ndarray: Array containing the predicted probabilities for each class.
    """
    assert not use_tta, "TTA not implemented"
    predict_fct = predict  # predict_tta if use_tta else predict

    config = Config(json.load(open(exp_folder + "config.json", "r")))

    model = define_model(
        config.name,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        use_gem=config.use_gem,
        reduce_stride=config.reduce_stride,
        verbose=(config.local_rank == 0),
        pretrained=False,
    )
    model = model.cuda().eval()

    preds = []
    for fold in config.selected_folds:
        print(f"\n- Fold {fold + 1}")

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=1)

        dataset = ClsDataset(
            df,
            transforms=get_transfos(augment=False, resize=config.resize),
        )

        pred, _ = predict_fct(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"],
            use_fp16=use_fp16,
        )

        if "target" in df.columns:
            acc = accuracy(df["target"].values, pred)
            print(f"\n -> Accuracy : {acc:.4f}")

        preds.append(pred)

    return np.mean(preds, 0)
