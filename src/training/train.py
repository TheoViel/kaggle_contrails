import gc
import time
import torch
import operator
import numpy as np
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders
from training.losses import ContrailLoss
from training.mix import Mixup, Cutmix
from training.optim import define_optimizer
from training.meter import SegmentationMeter

from util.torch import sync_across_gpus


def evaluate(
    model,
    val_loader,
    loss_config,
    loss_fct,
    use_fp16=False,
    distributed=False,
    world_size=0,
    local_rank=0,
):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_config (dict): Configuration parameters for the loss function.
        loss_fct (nn.Module): The loss function to compute the evaluation loss.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        world_size (int, optional): Number of processes in distributed training. Defaults to 0.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.

    Returns:
        val_loss (float or int): Average validation loss.
        dice (float or int): Dice score.
        acc (float or int): Accuracy score.
    """

    model.eval()
    val_losses = []

    meter = SegmentationMeter()

    with torch.no_grad():
        for img, y, y_aux in val_loader:
            img = img.cuda()
            y = y.cuda()
            y_aux = y_aux.cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(img)
                loss = loss_fct(y_pred.detach(), y_pred_aux, y, y_aux)

            val_losses.append(loss.detach())

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)
            if loss_config["activation_aux"] == "sigmoid":
                y_pred_aux = y_pred_aux.sigmoid()
            elif loss_config["activation_aux"] == "softmax":
                y_pred_aux = y_pred_aux.softmax(-1)

            meter.update(y, y_aux, y_pred, y_pred_aux)

    val_losses = torch.stack(val_losses)
    accs = torch.cat(meter.accs, 0)
    if distributed:
        val_losses = sync_across_gpus(val_losses, world_size)
        accs = sync_across_gpus(accs, world_size)
        torch.distributed.barrier()
    
    dices = {}
    
    for th in meter.thresholds:
        unions = meter.unions[th]
        intersections = meter.intersections[th]
        
        if distributed:
            unions = sync_across_gpus(unions, world_size)
            intersections = sync_across_gpus(intersections, world_size)
            torch.distributed.barrier()

        if local_rank == 0:
            dices[th] = (2 * intersections.sum() / unions.sum()).item()
            acc = accs.mean().item()

#     print(intersections.sum(), unions.sum())
    if local_rank == 0:
        acc = accs.mean().item()
        if not loss_config['aux_loss_weight']:
            acc = 0
        val_loss = np.nanmean(val_losses.cpu().numpy())
        return val_loss, dices, acc
    else:
        return 0, 0, 0


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    epochs=1,
    verbose_eval=1,
    use_fp16=False,
    distributed=False,
    local_rank=0,
    world_size=1,
    log_folder=None,
    run=None,
    fold=0,
    resume_step=1,
):
    """
    Train the model.

    Args:
        model (nn.Module): The main model to train.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        data_config (dict): Configuration parameters for data loading.
        loss_config (dict): Configuration parameters for the loss function.
        optimizer_config (dict): Configuration parameters for the optimizer.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        verbose_eval (int, optional): Number of steps for verbose evaluation. Defaults to 1.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        model_soup (bool, optional): Whether to save model weights for soup. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.
        world_size (int, optional): Number of processes in distributed training. Defaults to 1.
        log_folder (str, optional): Folder path for saving model weights. Defaults to None.
        run (neptune.Run, optional): Neptune run object for logging. Defaults to None.
        fold (int, optional): Fold number for tracking progress. Defaults to 0.
        resume_step (int, optional): Current training step for resuming training. Defaults to 1.

    Returns:
        preds (torch.Tensor or int): Predictions for the main task.
    """
    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        model,
        optimizer_config["name"],
        lr=optimizer_config["lr"],
        lr_encoder=optimizer_config["lr_encoder"],
        betas=optimizer_config["betas"],
        weight_decay=optimizer_config["weight_decay"],
    )

    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        val_bs=data_config["val_bs"],
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )

    # LR Scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    loss_fct = ContrailLoss(loss_config)

    if data_config["mix"] == "cutmix":
        mix = Cutmix(
            data_config["mix_alpha"],
            data_config["additive_mix"],
            data_config["num_classes"]
        )
    else:
        mix = Mixup(
            data_config["mix_alpha"],
            data_config["additive_mix"],
            data_config["num_classes"]
        )

    acc = 0
    step, step_ = 1, 1
    avg_losses, dices = [], {}
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        if distributed:
            try:
                train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

        for img, y, y_aux in train_loader:
            img = img.cuda()
            y = y.cuda()
            y_aux = y_aux.cuda()

            if np.random.random() < data_config["mix_proba"]:
                img, y, y_aux = mix(img, y, y_aux)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(img)

                loss = loss_fct(y_pred, y_pred_aux, y, y_aux)

            scaler.scale(loss).backward()
            avg_losses.append(loss.detach())

            scaler.unscale_(optimizer)

            if optimizer_config["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optimizer_config["max_grad_norm"]
                )

            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()

            model.zero_grad(set_to_none=True)

            if distributed:
                torch.cuda.synchronize()

            if scale == scaler.get_scale():
                scheduler.step()

            step += 1
            if (step % verbose_eval) == 0 or step - 1 >= epochs * len(train_loader):
                if 0 <= epochs * len(train_loader) - step < verbose_eval:
                    continue

                avg_losses = torch.stack(avg_losses)
                if distributed:
                    avg_losses = sync_across_gpus(avg_losses, world_size)
                avg_loss = avg_losses.cpu().numpy().mean()

                avg_val_loss, dices, acc = evaluate(
                    model,
                    val_loader,
                    loss_config,
                    loss_fct,
                    use_fp16=use_fp16,
                    distributed=distributed,
                    world_size=world_size,
                    local_rank=local_rank,
                )

                if local_rank == 0:
                    dt = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    step_ = step * world_size
                    
                    th, dice = max(dices.items(), key=operator.itemgetter(1))

                    s = f"Epoch {epoch:02d}/{epochs:02d} (step {step_:04d}) \t"
                    s = s + f"lr={lr:.1e} \t t={dt:.0f}s  \t loss={avg_loss:.3f}"
                    s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                    s = s + f"    dice@.5={dices[0.5]:.3f}" if dice else s
                    s = s + f"    dice={dice:.3f} (th={th:.2f})" if dice else s
                    s = s + f"    acc={acc:.3f}" if acc else s

                    print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step_ + resume_step)
                    run[f"fold_{fold}/train/loss"].log(avg_loss, step=step_ + resume_step)
                    run[f"fold_{fold}/train/lr"].log(lr, step=step_ + resume_step)
                    if not np.isnan(avg_val_loss):
                        run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step_ + resume_step)
                    run[f"fold_{fold}/val/dice"].log(dice, step=step_ + resume_step)
                    run[f"fold_{fold}/val/th"].log(th, step=step_ + resume_step)
                    run[f"fold_{fold}/val/dice@.5"].log(dices[0.5], step=step_ + resume_step)
                    run[f"fold_{fold}/val/acc"].log(acc, step=step_ + resume_step)

                start_time = time.time()
                avg_losses = []
                model.train()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    if distributed:
        torch.distributed.barrier()

    return dices, step_
