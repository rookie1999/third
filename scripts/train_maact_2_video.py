import argparse
import glob
import os
import pickle
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import timedelta


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from dataset.efficient_dataset import EfficientEpisodicDataset
from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
from dataset.utils_norm import get_norm_stats
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot, kl_divergence

from dataset.efficient_ma_dynamic_video_dataset import VideoBasedEfficientMADataset


def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--video', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., policy_epoch_500.ckpt)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from (used when resuming)')
    args = parser.parse_args()

    # 路径配置
    if args.video:
        # 视频模式：指向 episode 文件夹 (代码会自动找同级的 video 文件夹)
        DATA_DIR = r'F:\projects\lumos\data\20260109\episode'
    else:
        # 原始模式：指向包含全量数据的 hdf5 文件夹
        DATA_DIR = r'F:\projects\lumos\data\20260109'

    RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_maact")
    logger = setup_logger(RUN_DIR, name="MA_ACT")
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    mode_str = "VIDEO-Based" if args.video else "HDF5-RAM-Based"
    logger.info(f"🚀 MA-ACT Training Started! Mode: [{mode_str}] | Run ID: {RUN_NAME}")

    # 超参数配置
    NUM_EPOCHS = 1000
    BATCH_SIZE = 64
    LR = 1e-4
    LR_BACKBONE = 1e-5
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    SPEED_WEIGHT = 0.2

    CAMERA_NAMES = ['cam_high']
    MAIN_CAMERA_NAME = 'cam_high'
    N_OBS_STEPS = 2

    num_workers = 6 if args.video else 4

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))
    if len(dataset_path_list) == 0:
        logger.error(f"No HDF5 files found in {DATA_DIR}. Please check the path.")
        return

    # 统计信息 (Stats)
    if not os.path.exists(STATS_PATH):
        logger.info(f"Computing stats from {DATA_DIR}...")
        stats = get_norm_stats(DATA_DIR)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)
    else:
        logger.info(f"Loading stats from {STATS_PATH}...")
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)

    STATE_DIM = stats['qpos_mean'].shape[0]
    ACTION_DIM = stats['action_mean'].shape[0]

    if args.video:
        logger.info(f"Initializing Video MA-Dataset with {len(dataset_path_list)} episodes...")
        train_dataset = VideoBasedEfficientMADataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS  # 传入 n_obs_steps
        )
    else:
        logger.info(f"Initializing HDF5 MA-Dataset with {len(dataset_path_list)} episodes...")
        train_dataset = EfficientEpisodicDataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    # 4. 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = SpeedACTConfig(
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS,
        image_features={cam: (3, 480, 640) for cam in CAMERA_NAMES},
        main_camera=MAIN_CAMERA_NAME,
        robot_state_feature=(STATE_DIM,),
        action_feature=(ACTION_DIM,),
        use_optical_flow=True,
        feedforward_activation="relu",
        pre_norm=False,
        global_flow_size=128,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
    )
    policy = SpeedACT(config).to(device)

    param_groups = [
        {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"🔄 Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            # 注意：目前的 save 代码只保存了 state_dict，直接加载即可
            # 如果之前保存了 optimizer 状态，这里也可以加载 optimizer.load_state_dict(...)
            policy.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded weights successfully. Resuming from epoch {args.start_epoch}")
        else:
            logger.error(f"❌ Checkpoint file not found: {args.resume}")
            return

    best_loss = float('inf')
    train_losses = []
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    loss_func_speed = torch.nn.CrossEntropyLoss()

    total_start_time = time.time()

    for epoch in range(args.start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        policy.train()
        epoch_loss, epoch_l1, epoch_kl = 0, 0, 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            images_list, qpos, action, is_pad, speed_labels = data
            images_list = [x.to(device, non_blocking=True) for x in images_list]
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)
            speed_labels = speed_labels.to(device, non_blocking=True)

            norm_imgs = [(img - NORM_MEAN) / NORM_STD for img in images_list]

            batch_input = {
                "observation.state": qpos, "action": action, "action_is_pad": is_pad,
                "observation.images": norm_imgs, MAIN_CAMERA_NAME: norm_imgs[0]
            }

            pred_actions, (mu, logvar), speed_logits, flow_mask = policy(batch_input)

            all_l1 = F.l1_loss(pred_actions, action, reduction='none')

            # l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            n_valid = (~is_pad).sum()
            if n_valid > 0:
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (n_valid * ACTION_DIM + 1e-6)
            else:
                l1 = torch.tensor(0.0, device=device)

            total_kld, _, _ = kl_divergence(mu, logvar)
            kl_loss = total_kld[0]

            speed_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if speed_logits is not None and flow_mask.sum() > 0:
                valid_logits = speed_logits[flow_mask]
                valid_labels = speed_labels[flow_mask]
                speed_loss = loss_func_speed(valid_logits, valid_labels)

            loss = l1 + KL_WEIGHT * kl_loss + SPEED_WEIGHT * speed_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_kl += kl_loss.item()


        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Time & Log
        epoch_dur = time.time() - epoch_start
        eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

        logger.info(
            f"Epoch {epoch:04d} | Loss: {avg_loss:.5f} (L1: {epoch_l1 / len(train_loader):.5f}, KL: {epoch_kl / len(train_loader):.5f}) | Time: {epoch_dur:.1f}s | ETA: {eta}")

        # Save
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch}.ckpt")
            torch.save(policy.state_dict(), save_path)
            save_train_loss_plot(RUN_DIR, train_losses, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(CKPT_DIR, "policy_best.ckpt"))
            logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    logger.info("Training Done!")


if __name__ == '__main__':
    main()