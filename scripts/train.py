import glob
import os
import pickle
import sys
import time
import torch
from torch.utils.data import DataLoader
from datetime import timedelta

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 引入 ACT 相关模块
from policy.act.policy import ACTPolicy
from dataset.efficient_dataset import EfficientEpisodicDataset
from dataset.utils_norm import get_norm_stats
# 引入公共工具 (确保 scripts/utils_train.py 存在)
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot


def main():
    # 1. 目录与日志 (使用 logs_act 目录)
    RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_act")
    logger = setup_logger(RUN_DIR, name="ACT")
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    logger.info(f"🚀 Standard ACT Training Started! Run ID: {RUN_NAME}")

    # 2. 核心配置
    DATA_DIR = r'F:\projects\lumos\data\20260109'
    NUM_EPOCHS = 500
    BATCH_SIZE = 16  # ACT 较轻量，可能不需要太大 Batch 或累积
    LR = 1e-4
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0

    CAMERA_NAMES = ['cam_high']
    STATE_DIM = 8  # 3d坐标 + 四元数 + 夹爪
    ACTION_DIM = 8

    # 3. 数据集
    if not os.path.exists(STATS_PATH):
        logger.info(f"Computing stats...")
        stats = get_norm_stats(DATA_DIR)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)
    else:
        logger.info(f"Loading stats...")
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))
    train_dataset = EfficientEpisodicDataset(
        dataset_path_list, stats, camera_names=CAMERA_NAMES,
        chunk_size=CHUNK_SIZE
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=4, prefetch_factor=2
    )

    # 4. 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ACTPolicy 的参数构造
    args_override = {
        'kl_weight': KL_WEIGHT,
        'chunk_size': CHUNK_SIZE,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'nheads': 8,
        'enc_layers': 4,
        'dec_layers': 1,
        'n_decoder_layers': 1,  # 兼容不同命名习惯
        'camera_names': CAMERA_NAMES,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'masks': False,
        'dilation': False,
        'dropout': 0.1,
        'pre_norm': False,
        'num_queries': CHUNK_SIZE,
    }

    policy = ACTPolicy(args_override).to(device)
    optimizer = policy.configure_optimizers()  # ACTPolicy 通常自带这个方法

    # 5. 训练循环
    best_loss = float('inf')
    train_losses = []

    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            # Dataset 返回: images_list, qpos, action, is_pad
            # images_list[0] shape: (B, C, H, W) 因为 N_OBS_STEPS=1
            image_tensor, qpos, action, is_pad = data

            # 移动到 GPU
            image = image_tensor.to(device, non_blocking=True)  # (B, C, H, W)
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            # Forward (ACTPolicy 内部计算 Loss)
            loss_dict = policy(qpos, image, action, is_pad)

            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Log
        epoch_dur = time.time() - epoch_start
        eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

        logger.info(f"Epoch {epoch:04d} | Loss: {avg_loss:.5f} | Time: {epoch_dur:.1f}s | ETA: {eta}")

        # Save
        if epoch % 500 == 0:
            torch.save(policy.state_dict(), os.path.join(CKPT_DIR, f"policy_epoch_{epoch}.ckpt"))
            save_train_loss_plot(RUN_DIR, train_losses, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), os.path.join(CKPT_DIR, "policy_best.ckpt"))
            logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    logger.info("Training Done!")


if __name__ == '__main__':
    main()