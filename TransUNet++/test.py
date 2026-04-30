
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from datasets.dataset_synapse import Synapse_dataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


# ============================
# ARGUMENTS
# ============================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path",        type=str, default="../data/Synapse/test_npz")
parser.add_argument("--dataset",          type=str, default="Synapse")
parser.add_argument("--num_classes",      type=int, default=2)
parser.add_argument("--list_dir",         type=str, default="./lists/lists_Synapse")
parser.add_argument("--img_size",         type=int, default=224)
parser.add_argument("--n_skip",           type=int, default=3)
parser.add_argument("--vit_name",         type=str, default="R50-ViT-B_16-Plus")
parser.add_argument("--test_save_dir",    type=str, default="../predictions")
parser.add_argument("--deterministic",    type=int, default=1)
parser.add_argument("--seed",             type=int, default=1234)
parser.add_argument("--vit_patches_size", type=int, default=16)
parser.add_argument("--snapshot_path",    type=str, default="",
                    help="Full path to best_model.pth")
args = parser.parse_args()


# ============================
# METRICS
# ============================
def compute_metrics(pred, target):
    """
    pred   : (H, W) long tensor — argmax class map
    target : (H, W) long tensor — ground truth
    Both must be the same spatial size (enforced in inference()).
    """
    pred   = pred.view(-1)
    target = target.view(-1)
    TP = ((pred == 1) & (target == 1)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()
    precision = (TP / (TP + FP + 1e-8)).item()
    recall    = (TP / (TP + FN + 1e-8)).item()
    iou       = (TP / (TP + FP + FN + 1e-8)).item()
    dice      = (2 * TP / (2 * TP + FP + FN + 1e-8)).item()
    return precision, recall, iou, dice


# ============================
# SAVE IMAGE
# ============================
def save_prediction(image, pred, gt, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    image = image.squeeze().cpu().numpy()
    pred  = pred.squeeze().cpu().numpy()
    gt    = gt.squeeze().cpu().numpy()
    if len(image.shape) == 3:
        image = image[0]
    image  = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image  = (image * 255).astype(np.uint8)
    pred   = (pred  * 255).astype(np.uint8)
    gt     = (gt    * 255).astype(np.uint8)
    concat = np.concatenate([image, pred, gt], axis=1)
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), concat)


# ============================
# INFERENCE
# ============================
def inference(args, model, device, test_save_path):
    db_test = Synapse_dataset(
        base_dir=args.data_path,
        list_dir=args.list_dir,
        split="test",
        img_size=args.img_size,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    logging.info(f"{len(testloader)} test samples")

    model.eval()
    total_precision = total_recall = total_iou = total_dice = 0.0
    count = 0
    y_true, y_scores = [], []

    for sampled_batch in tqdm(testloader, total=len(testloader)):
        image     = sampled_batch["image"].to(device)   # (1, C, H, W)
        label     = sampled_batch["label"].to(device)   # (1, H, W)
        case_name = sampled_batch["case_name"][0]

        target_size = label.shape[1:]   # (H, W) of the ground-truth label

        with torch.no_grad():
            outputs = model(image)

            # Unpack tuple from nested decoder (e.g. R50-ViT-B_16-Plus)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]    # (1, n_classes, h, w)  — may be smaller than label

            # ── KEY FIX: resize logits to match label spatial size ──────────
            # The nested decoder final output can be 112x112 while label is 224x224.
            # Always align before softmax/argmax so metrics are computed correctly.
            if outputs.shape[2:] != target_size:
                outputs = F.interpolate(
                    outputs,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            # ────────────────────────────────────────────────────────────────

            probs = torch.softmax(outputs, dim=1)[:, 1, :, :]   # (1, H, W)
            pred  = torch.argmax(outputs, dim=1)                 # (1, H, W)

        # pred  : (1, H, W),  label : (1, H, W)  — same size guaranteed
        y_true.extend(label.cpu().numpy().flatten())
        y_scores.extend(probs.cpu().numpy().flatten())

        precision, recall, iou, dice = compute_metrics(
            pred.squeeze(0), label.squeeze(0)
        )
        total_precision += precision
        total_recall    += recall
        total_iou       += iou
        total_dice      += dice
        count           += 1

        logging.info(
            f"{case_name} | Dice={dice:.4f}  IoU={iou:.4f}  "
            f"Precision={precision:.4f}  Recall={recall:.4f}"
        )
        save_prediction(image, pred, label, test_save_path, case_name)

    # ── Final summary ──────────────────────────────────────────────────────
    logging.info("====== FINAL RESULTS ======")
    logging.info(f"Dice:      {total_dice      / count:.4f}")
    logging.info(f"IoU:       {total_iou       / count:.4f}")
    logging.info(f"Precision: {total_precision / count:.4f}")
    logging.info(f"Recall:    {total_recall    / count:.4f}")

    print()
    print("====== FINAL RESULTS ======")
    print(f"  Dice      : {total_dice      / count:.4f}")
    print(f"  IoU       : {total_iou       / count:.4f}")
    print(f"  Precision : {total_precision / count:.4f}")
    print(f"  Recall    : {total_recall    / count:.4f}")

    # ── ROC & PR curves ────────────────────────────────────────────────────
    y_true_np   = np.array(y_true)
    y_scores_np = np.array(y_scores)

    if np.sum(y_true_np) > 0:
        fpr, tpr, _ = roc_curve(y_true_np, y_scores_np)
        roc_auc     = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(test_save_path, "ROC_curve.png"))
        plt.close()

        precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_scores_np)
        ap = average_precision_score(y_true_np, y_scores_np)

        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig(os.path.join(test_save_path, "PR_curve.png"))
        plt.close()

        print(f"  ROC AUC           : {roc_auc:.4f}")
        print(f"  Average Precision : {ap:.4f}")
        print(f"  Curves saved → {test_save_path}")
    else:
        print("No positive samples — ROC/PR curves skipped.")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark     = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark     = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model config ───────────────────────────────────────────────────────
    if args.vit_name not in CONFIGS_ViT_seg:
        raise ValueError(
            f"Model '{args.vit_name}' not in CONFIGS. "
            f"Available: {list(CONFIGS_ViT_seg.keys())}"
        )

    config_vit           = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip    = args.n_skip

    if any(k in args.vit_name for k in ("R50", "ConvNeXt", "EfficientNet")):
        config_vit.patches.grid = (
            args.img_size // args.vit_patches_size,
            args.img_size // args.vit_patches_size,
        )

    net = ViT_seg(
        config_vit,
        img_size=args.img_size,
        num_classes=config_vit.n_classes,
    ).to(device)

    # ── Load weights ───────────────────────────────────────────────────────
    snapshot = args.snapshot_path

    if not snapshot:
        exp_name = (
            f"TU_{args.vit_name.replace('-', '_')}"
            f"_skip{args.n_skip}_epo15_bs8"
        )
        snapshot = os.path.join(
            "/kaggle/working/project-transunet/model",
            f"TU_Synapse{args.img_size}",
            exp_name,
            "best_model.pth",
        )

    if os.path.exists(snapshot):
        print(f"Loading model weights from {snapshot}")
        net.load_state_dict(torch.load(snapshot, map_location=device))
    else:
        print(f"WARNING: weights not found at {snapshot} — running with random weights")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    test_save_path = args.test_save_dir
    os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, device, test_save_path)
