
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from medpy.metric.binary import hd95
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_dice(pred, target):
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    intersection = (pred * target).sum().float()
    union = pred.sum() + target.sum()
    return ((2.0 * intersection + 1e-5) / (union + 1e-5)).item()


def compute_iou(pred, target):
    pred   = torch.argmax(torch.softmax(pred, dim=1), dim=1).bool()
    target = target.bool()
    intersection = (pred & target).float().sum()
    union        = (pred | target).float().sum()
    return ((intersection + 1e-5) / (union + 1e-5)).item()


def compute_hd95(pred, target):
    pred      = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    pred_np   = pred.cpu().numpy().astype(np.bool_)
    target_np = target.cpu().numpy().astype(np.bool_)
    try:
        return float(hd95(pred_np, target_np))
    except Exception:
        return 0.0


def compute_precision_recall(pred, target):
    pred   = torch.argmax(torch.softmax(pred, dim=1), dim=1).view(-1)
    target = target.view(-1)
    TP = ((pred == 1) & (target == 1)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()
    precision = (TP / (TP + FP + 1e-8)).item()
    recall    = (TP / (TP + FN + 1e-8)).item()
    return precision, recall


# ── Deep-supervision loss ──────────────────────────────────────────────────────

def deep_supervision_loss(aux_logits, label_batch, ce_loss_fn, dice_loss_fn,
                           weights=None):
    if weights is None:
        weights = [1.0 / len(aux_logits)] * len(aux_logits)
    loss = 0.0
    for w, aux in zip(weights, aux_logits):
        if aux.shape[2:] != label_batch.shape[1:]:
            aux = torch.nn.functional.interpolate(
                aux, size=label_batch.shape[1:],
                mode="bilinear", align_corners=False,
            )
        l_ce   = ce_loss_fn(aux, label_batch.long())
        l_dice = dice_loss_fn(aux, label_batch, softmax=True)
        loss  += w * (0.5 * l_ce + 0.5 * l_dice)
    return loss


# ── Trainer ───────────────────────────────────────────────────────────────────

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    os.makedirs(snapshot_path, exist_ok=True)

    # ── Logger: file only (no console spam) ──────────────────────────────────
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(os.path.join(snapshot_path, "log.txt"))
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    # Console handler kept but will only print epoch summaries (not per-iter)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    ch.setLevel(logging.WARNING)   # suppress INFO from console — epochs only via tqdm
    logger.addHandler(ch)

    logging.info("========== TRAINING START ==========")
    logging.info(str(args))

    # Dataset
    batch_size  = args.batch_size * args.n_gpu
    num_classes = args.num_classes
    base_lr     = args.base_lr

    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ]),
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=lambda wid: random.seed(args.seed + wid),
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss_fn   = CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr, momentum=0.9, weight_decay=0.0001,
    )

    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    use_deep_supervision = getattr(args, "use_deep_supervision", False)
    ds_weights     = [0.2, 0.3, 0.5]
    iter_num       = 0
    max_epoch      = args.max_epochs
    max_iterations = max_epoch * len(trainloader)
    best_loss      = float("inf")

    print(f"[INFO] Dataset size : {len(db_train)}")
    print(f"[INFO] Batches/epoch: {len(trainloader)}")
    print(f"[INFO] Total epochs : {max_epoch}")
    print(f"[INFO] Logs → {os.path.join(snapshot_path, 'log.txt')}")
    print()

    # ── Epoch loop ────────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(max_epoch), desc="Training", unit="epoch",
                     dynamic_ncols=True)

    for epoch_num in epoch_bar:
        model.train()
        epoch_loss = 0.0
        total_dice = total_iou = total_precision = total_recall = total_hd = 0.0
        count = 0

        for sampled_batch in trainloader:
            image_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()

            output = model(image_batch)

            if isinstance(output, (tuple, list)):
                logits, aux_logits = output[0], output[1]
            else:
                logits, aux_logits = output, None

            if logits.shape[2:] != label_batch.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=label_batch.shape[1:],
                    mode="bilinear", align_corners=False,
                )

            loss = (0.5 * ce_loss_fn(logits, label_batch.long()) +
                    0.5 * dice_loss_fn(logits, label_batch, softmax=True))

            if use_deep_supervision and aux_logits is not None:
                loss_ds = deep_supervision_loss(
                    aux_logits, label_batch, ce_loss_fn, dice_loss_fn, ds_weights)
                loss = loss + 0.4 * loss_ds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr_

            iter_num   += 1
            epoch_loss += loss.item()

            dice              = compute_dice(logits, label_batch)
            iou               = compute_iou(logits, label_batch)
            hd_val            = compute_hd95(logits, label_batch)
            precision, recall = compute_precision_recall(logits, label_batch)

            total_dice      += dice
            total_iou       += iou
            total_hd        += hd_val
            total_precision += precision
            total_recall    += recall
            count           += 1

            # All per-iter detail goes to log file only
            logging.info(
                f"iter {iter_num:05d} | loss={loss.item():.4f}  "
                f"dice={dice:.4f}  iou={iou:.4f}  "
                f"prec={precision:.4f}  recall={recall:.4f}  "
                f"hd95={hd_val:.2f}  lr={lr_:.6f}"
            )

            writer.add_scalar("iter/loss",      loss.item(), iter_num)
            writer.add_scalar("iter/dice",      dice,        iter_num)
            writer.add_scalar("iter/iou",       iou,         iter_num)
            writer.add_scalar("iter/hd95",      hd_val,      iter_num)
            writer.add_scalar("iter/precision", precision,   iter_num)
            writer.add_scalar("iter/recall",    recall,      iter_num)
            writer.add_scalar("iter/lr",        lr_,         iter_num)

        # ── Epoch summary ─────────────────────────────────────────────────────
        n          = len(trainloader)
        epoch_loss /= n
        avg_dice   = total_dice      / count
        avg_iou    = total_iou       / count
        avg_hd     = total_hd        / count
        avg_prec   = total_precision / count
        avg_rec    = total_recall    / count

        writer.add_scalar("epoch/loss",      epoch_loss, epoch_num)
        writer.add_scalar("epoch/dice",      avg_dice,   epoch_num)
        writer.add_scalar("epoch/iou",       avg_iou,    epoch_num)
        writer.add_scalar("epoch/hd95",      avg_hd,     epoch_num)
        writer.add_scalar("epoch/precision", avg_prec,   epoch_num)
        writer.add_scalar("epoch/recall",    avg_rec,    epoch_num)

        summary = (
            f"Epoch {epoch_num+1:03d}/{max_epoch} | "
            f"loss={epoch_loss:.4f}  dice={avg_dice:.4f}  "
            f"iou={avg_iou:.4f}  hd95={avg_hd:.2f}  "
            f"prec={avg_prec:.4f}  rec={avg_rec:.4f}"
        )

        # Update tqdm bar with key metrics
        epoch_bar.set_postfix({
            "loss": f"{epoch_loss:.4f}",
            "dice": f"{avg_dice:.4f}",
            "iou":  f"{avg_iou:.4f}",
        })

        # Print epoch summary to console + log file
        tqdm.write(summary)
        logging.info(summary)

        # Best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(snapshot_path, "best_model.pth")
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, save_path)
            msg = f"  ✓ Best model saved (loss={best_loss:.4f}) → {save_path}"
            tqdm.write(msg)
            logging.info(msg)

        # Periodic checkpoint every 10 epochs
        if (epoch_num + 1) % 10 == 0:
            ckpt = os.path.join(snapshot_path, f"epoch_{epoch_num+1:03d}.pth")
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save(state, ckpt)
            msg = f"  ✓ Checkpoint saved → {ckpt}"
            tqdm.write(msg)
            logging.info(msg)

    writer.close()
    logging.info("========== TRAINING FINISHED ==========")
    print("\n========== TRAINING FINISHED ==========")
    return "Training Finished!"
