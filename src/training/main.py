import gc
import glob
import json
import torch
import operator
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models import define_model

from data.dataset import ContrailDataset
from data.transforms import get_transfos

from util.torch import seed_everything, count_parameters, save_model_weights


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (neptune.Run): Nepture run. Defaults to None.

    Returns:
        dict: Dice scores at different thresholds.
    """
    transforms = get_transfos(
        strength=config.aug_strength,
        crop=config.pretrain if hasattr(config, "pretrain") else False
    )

    train_dataset = ContrailDataset(
        df_train,
        transforms=transforms,
        use_soft_mask=config.use_soft_mask,
        use_shape_descript=config.use_shape_descript,
        use_pl_masks=config.use_pl_masks,
        frames=config.frames,
        use_ext_data=config.use_ext_data if hasattr(config, "use_ext_data") else False,
        aug_strength=config.aug_strength,
    )

    val_dataset = ContrailDataset(
        df_val,
        transforms=get_transfos(augment=False),
        frames=config.frames,
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[
                0
            ]
    else:
        pretrained_weights = None

    model = define_model(
        config.decoder_name,
        config.encoder_name,
        num_classes=config.num_classes,
        n_channels=config.n_channels,
        pretrained_weights=pretrained_weights,
        reduce_stride=config.reduce_stride,
        upsample=config.upsample if hasattr(config, "upsample") else False,
        use_pixel_shuffle=config.use_pixel_shuffle,
        use_hypercolumns=config.use_hypercolumns,
        center=config.center,
        use_cls=config.loss_config['aux_loss_weight'] > 0,
        frames=config.frames,
        use_lstm=config.use_lstm,
        bidirectional=config.bidirectional,
        use_cnn=config.use_cnn,
        kernel_size=config.kernel_size,
        use_transfo=config.use_transfo,
        two_layers=config.two_layers if hasattr(config, "two_layers") else False,
        verbose=(config.local_rank == 0),
    ).cuda()

    if config.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters\n")

    dices = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.decoder_name}_{config.encoder_name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return dices


def k_fold(config, df, df_extra=None, log_folder=None, run=None):
    """
    Trains a k-fold.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (pandas dataframe or None, optional): Extra metadata. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (None or Nepture run): Nepture run. Defaults to None.

    Returns:
        dict: Dice scores at different thresholds.
    """
    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left")

    scores = []
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            train_idx = list(df[df["fold"] != fold].index)
            val_idx = list(df[df["fold"] == fold].index)

            df_train = df.iloc[train_idx].reset_index(drop=True)

            if hasattr(config, "pretrain"):
                if config.pretrain:
                    df_train = pd.read_csv('../output/df_goes16_may.csv')

            df_val = df.iloc[val_idx].reset_index(drop=True)

            if len(df) <= 1000:
                df_train, df_val = df, df

            dices = train(
                config, df_train, df_val, fold, log_folder=log_folder, run=run
            )
            scores.append(dices)

    if config.local_rank == 0 and len(config.selected_folds):
        dices = {th: np.mean([dice[th] for dice in scores]) for th in scores[0].keys()}
        th, dice = max(dices.items(), key=operator.itemgetter(1))

        print(f"\n\n -> CV Dice : {dice:.3f}  -  th : {th:.2f}")

        if log_folder is not None:
            json.dump(dices, open(log_folder + "dices.json", "w"))

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")
            run["global/cv"] = dice
            run["global/th"] = th

    if config.fullfit:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            train(
                config,
                df,
                df.tail(100).reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()

    return dices
