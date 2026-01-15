import argparse
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
from dataset.utils_norm import get_norm_stats
# 引入公共工具
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot

# --- [修改点 1] 引入两个 Dataset 类 ---
from dataset.efficient_dataset import EfficientEpisodicDataset

try:
    # 尝试引入视频 Dataset，如果文件不存在则忽略（防止报错）
    from dataset.efficient_video_dataset import VideoBasedEfficientDataset
except ImportError:
    VideoBasedEfficientDataset = None
    print("⚠️ Warning: efficient_video_dataset.py not found. Video mode unavailable.")


def main():
    parser = argparse.ArgumentParser(description="ACT Training Script")
    parser.add_argument('--video', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., policy_epoch_500.ckpt)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from (used when resuming)')
    args = parser.parse_args()


    # 数据路径配置
    if args.video:
        # 视频模式下，建议指向 episode 文件夹，代码会自动找同级的 video 文件夹
        # 结构示例: .../20260109/episode/*.hdf5 和 .../20260109/video/*.mp4
        DATA_DIR = r'F:\projects\lumos\train_data\episode'
    else:
        # 原始模式，指向包含完整数据的 HDF5 文件夹
        DATA_DIR = r'F:\projects\lumos\data\20260109'

    RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_act")
    logger = setup_logger(RUN_DIR, name="ACT")
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    mode_str = "VIDEO-Based" if args.video else "HDF5-RAM-Based"
    logger.info(f"🚀 ACT Training Started! Mode: [{mode_str}] | Run ID: {RUN_NAME}")

    # 超参数
    NUM_EPOCHS = 500
    BATCH_SIZE = 16
    LR = 1e-4
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    CAMERA_NAMES = ['cam_high']
    STATE_DIM = 8
    ACTION_DIM = 8

    # 硬件配置
    # num_workers = 8 if args.video else 4  # 视频解码需要更多 CPU 线程
    num_workers = 4


    # ==========================
    # 2. 数据集准备
    # ==========================
    # 无论哪种模式，我们都先获取 .hdf5 文件列表作为索引
    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))
    if len(dataset_path_list) == 0:
        logger.error(f"No HDF5 files found in {DATA_DIR}. Check your path!")
        return

    # 统计数据 (Stats) 计算/加载
    if not os.path.exists(STATS_PATH):
        logger.info(f"Computing stats from: {DATA_DIR} ...")
        stats = get_norm_stats(DATA_DIR)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)
    else:
        logger.info(f"Loading stats from: {STATS_PATH}")
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)

    if args.video:
        if VideoBasedEfficientDataset is None:
            raise ImportError("Cannot use LOAD_FROM_VIDEO=True because efficient_video_dataset.py is missing.")

        logger.info("Initializing VideoBasedEfficientDataset (Reading MP4s)...")
        train_dataset = VideoBasedEfficientDataset(
            dataset_path_list,
            stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE
        )
    else:
        logger.info("Initializing EfficientEpisodicDataset (Reading HDF5 images)...")
        train_dataset = EfficientEpisodicDataset(
            dataset_path_list,
            stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE
        )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    # ==========================
    # 3. 模型初始化
    # ==========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_override = {
        'kl_weight': KL_WEIGHT,
        'chunk_size': CHUNK_SIZE,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'nheads': 8,
        'enc_layers': 4,
        'dec_layers': 1,
        'n_decoder_layers': 1,
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
    optimizer = policy.configure_optimizers()

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

    # ==========================
    # 4. 训练循环 (完全复用，无需修改)
    # ==========================
    best_loss = float('inf')
    train_losses = []
    total_start_time = time.time()

    for epoch in range(args.start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            # Dataset 接口一致: image_tensor, qpos, action, is_pad
            image_tensor, qpos, action, is_pad = data

            image = image_tensor.to(device, non_blocking=True)
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            loss_dict = policy(qpos, image, action, is_pad)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        epoch_dur = time.time() - epoch_start
        eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

        logger.info(f"Epoch {epoch:04d} | Loss: {avg_loss:.5f} | Time: {epoch_dur:.1f}s | ETA: {eta}")

        if epoch % 50 == 0:
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