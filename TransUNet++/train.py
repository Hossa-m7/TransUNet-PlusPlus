
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz')
parser.add_argument('--dataset', type=str, default='Synapse')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')

parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=24)

parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)

parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--img_size', type=int, default=224)

parser.add_argument('--seed', type=int, default=1234)

parser.add_argument('--n_skip', type=int, default=2)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')

args = parser.parse_args()

# =========================
# Main
# =========================
if __name__ == "__main__":

    print("🔥 NEW TRAIN.PY RUNNING")  # Debug (VERY IMPORTANT)

    # -------------------------
    # Logging
    # -------------------------
    os.makedirs("../log", exist_ok=True)
    log_file = "../log/train_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("===== TRAINING START =====")

    # -------------------------
    # Deterministic behavior
    # -------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    # -------------------------
    # Seeds
    # -------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------
    # Config
    # -------------------------
    if args.vit_name not in CONFIGS_ViT_seg:
        raise ValueError(f"{args.vit_name} not found in CONFIGS")

    config_vit = CONFIGS_ViT_seg[args.vit_name]

    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    # 🔥 CRITICAL: disable ANY pretrained loading
    config_vit.pretrained_path = None

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Model
    # -------------------------
    net = ViT_seg(
        config_vit,
        img_size=args.img_size
    ).to(device)

    logging.info("✅ Model initialized (ConvNeXt + ViT)")

    # -------------------------
    # Snapshot path
    # -------------------------
    args.exp = 'TU_' + args.dataset + str(args.img_size)

    snapshot_path = f"../model/{args.exp}/TU_ConvNeXt"
    snapshot_path += "_skip" + str(args.n_skip)
    snapshot_path += "_epo" + str(args.max_epochs)
    snapshot_path += "_bs" + str(args.batch_size)

    os.makedirs(snapshot_path, exist_ok=True)

    logging.info(f"Snapshot path: {snapshot_path}")

    # -------------------------
    # Trainer
    # -------------------------
    trainer = {'Synapse': trainer_synapse}

    logging.info("🚀 Starting training...")

    trainer[args.dataset](args, net, snapshot_path)

    logging.info("===== TRAINING FINISHED =====")





