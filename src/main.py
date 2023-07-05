import os
import time
import torch
import warnings
import argparse
import pandas as pd

from data.preparation import prepare_data
from util.torch import init_distributed
from util.logger import create_logger, save_config, prepare_log_folder, init_neptune

from params import DATA_PATH


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold number",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """

    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Data
    processed_folder = "false_color/"
    size = 256
    aug_strength = 3
    use_soft_mask = True

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0] #, 1, 2, 3]

    # Model
    encoder_name = "convnextv2_nano"  # tf_efficientnetv2_s seresnext50_32x4d efficientnetv2_rw_t convnextv2_tiny convnextv2_nano
    decoder_name = "Unet"

    pretrained_weights = None
    reduce_stride = 2
    use_pixel_shuffle = False
    use_hypercolumns = False
    center = "none"
    n_channels = 3
    num_classes = 1

    # Training
    loss_config = {
        "name": "lovasz_bce",  # bce lovasz_focal lovasz focal
        "smoothing": 0.,
        "activation": "sigmoid",
        "aux_loss_weight": 0.,
        "activation_aux": "sigmoid",
        "ousm_k": 0,
    }

    data_config = {
        "batch_size": 8,
        "val_bs": 16,
        "mix": "cutmix",
        "mix_proba": 0.5,
        "mix_alpha": 5,
        "additive_mix": True,
        "num_classes": num_classes
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 2e-4,
        "lr_encoder": 2e-4,
        "warmup_prop": 0.05,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.05,
    }

    epochs = 40
    two_stage = False

    use_fp16 = True
    model_soup = False

    verbose = 1
    verbose_eval = 200

    fullfit = False  # len(selected_folds) == 4
    n_fullfit = 1


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = args.fold if args.fold > -1 else args.device
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    if not log_folder:
        from params import LOG_PATH

        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH)

    if args.model:
        config.name = args.model

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    df = prepare_data(DATA_PATH, Config.processed_folder)

    try:
        print(torch_performance_linter)  # noqa
        if config.local_rank == 0:
            print("Using TPL\n")
        run = None
        config.epochs = 1
        log_folder = None
        df = df.head(10000)
    except Exception:
        run = None
        if config.local_rank == 0:
            run = init_neptune(Config, log_folder)

            if args.fold > -1:
                config.selected_folds = [args.fold]
                create_logger(directory=log_folder, name=f"logs_{args.fold}.txt")
            else:
                create_logger(directory=log_folder, name="logs.txt")

            save_config(config, log_folder + "config.json")

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Decoder {config.decoder_name}")
        print(f"- Encoder {config.encoder_name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    from training.main import k_fold
#     df = df.head(1000)
    k_fold(Config, df, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
