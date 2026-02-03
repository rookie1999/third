import argparse
import glob
import os
import pickle
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import timedelta

# 获取当前路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 引入项目模块
from dataset.efficient_dataset import EfficientEpisodicDataset
from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
from dataset.utils_norm import get_norm_stats
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot, kl_divergence
from dataset.efficient_ma_dynamic_video_dataset import VideoBasedEfficientMADataset


# DDP 初始化函数
def ddp_setup():
    # 从环境变量获取 local_rank (torchrun 会自动设置)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # 如果不是 DDP 启动，回退到单卡模式或报错
        print("Not using DDP. Initializing single GPU...")
        return 0, 0, 1


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="ACT Training Script (DDP)")
    parser.add_argument('--video', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., policy_epoch_500.ckpt)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from')
    args = parser.parse_args()

    rank, local_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")

    # 路径配置
    if args.video:
        DATA_DIR = r'F:\projects\lumos\data\20260109\episode'
    else:
        DATA_DIR = r'F:\projects\lumos\data\20260109'

    # --- 2. 仅在主进程 (Rank 0) 初始化日志和目录 ---
    logger = None
    RUN_DIR, CKPT_DIR, RUN_NAME = None, None, None
    STATS_PATH = None

    if rank == 0:
        # 修改：日志根目录
        RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_maact")
        logger = setup_logger(RUN_DIR, name="MA_ACT")
        STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

        mode_str = "VIDEO-Based" if args.video else "HDF5-RAM-Based"
        logger.info(
            f"🚀 MA-ACT DDP Training Started! Mode: [{mode_str}] | World Size: {world_size} | Run ID: {RUN_NAME}")

    # 超参数配置
    NUM_EPOCHS = 1000
    BATCH_SIZE = 64  # 注意：这是单卡 Batch Size，总 Batch Size = 64 * GPU数量
    LR = 1e-5
    LR_BACKBONE = 1e-5
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    SPEED_WEIGHT = 0.2

    CAMERA_NAMES = ['cam_high']
    MAIN_CAMERA_NAME = 'cam_high'
    N_OBS_STEPS = 2

    # 多卡下 num_workers 分配给每个进程
    num_workers = int((8 if args.video else 4) / world_size)

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))
    if len(dataset_path_list) == 0:
        if rank == 0: logger.error(f"No HDF5 files found in {DATA_DIR}. Please check the path.")
        return

    # --- 3. 统计信息处理 (同步) ---
    # 定义 STATS_PATH 供所有进程读取（需要确保所有进程路径一致）
    # 简单处理：我们假设大家都在同一台机器上，路径相同。如果是多机，需要共享存储或广播 stats。
    # 这里我们重新构造 STATS_PATH 变量给非 Rank 0 进程
    STATS_PATH = os.path.join(DATA_DIR, 'dataset_stats_cache.pkl')

    if rank == 0:
        if not os.path.exists(STATS_PATH):
            logger.info(f"Computing stats from {DATA_DIR}...")
            stats = get_norm_stats(DATA_DIR)
            with open(STATS_PATH, 'wb') as f:
                pickle.dump(stats, f)
        else:
            logger.info(f"Loading stats from {STATS_PATH}...")

    # 等待 Rank 0 完成 stats 计算
    dist.barrier()

    with open(STATS_PATH, 'rb') as f:
        stats = pickle.load(f)

    STATE_DIM = stats['qpos_mean'].shape[0]
    ACTION_DIM = stats['action_mean'].shape[0]

    # --- 4. 数据集与 Sampler ---
    if args.video:
        if rank == 0: logger.info(f"Initializing Video MA-Dataset with {len(dataset_path_list)} episodes...")
        train_dataset = VideoBasedEfficientMADataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS
        )
    else:
        if rank == 0: logger.info(f"Initializing HDF5 MA-Dataset with {len(dataset_path_list)} episodes...")
        train_dataset = EfficientEpisodicDataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS
        )

    # 关键：使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 使用 Sampler 时必须为 False
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    # --- 5. 模型初始化与 DDP ---
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
        num_speed_categories=3,
    )

    policy = SpeedACT(config).to(device)

    # 可选：转换 BatchNorm 为 SyncBatchNorm (提升多卡训练时的 BN 精度)
    policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy)

    # 封装 DDP
    # find_unused_parameters=True 防止某些分支（如光流部分）未被执行时报错，但会轻微降低速度。
    # 如果确定所有参数都参与计算，可以设为 False。
    policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)

    # 优化器配置
    # 注意：DDP 封装后，参数名会增加 "module." 前缀，但 "backbone" 字符串匹配依然有效
    param_groups = [
        {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # 加载 Checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0: logger.info(f"🔄 Resuming training from checkpoint: {args.resume}")
            # map_location 必须指定为当前 device
            checkpoint = torch.load(args.resume, map_location=device)
            # DDP 模型的 state_dict key 带有 "module."，如果加载的是单卡权重，需要处理 key
            # 这里假设加载的是之前的 DDP 权重。如果是单卡权重，PyTorch 有时会自动处理，或者手动增加 module. 前缀
            policy.load_state_dict(checkpoint)
            if rank == 0: logger.info(f"✅ Loaded weights successfully. Resuming from epoch {args.start_epoch}")
        else:
            if rank == 0: logger.error(f"❌ Checkpoint file not found: {args.resume}")
            return

    best_loss = float('inf')
    train_losses = []

    # 归一化参数移至 device
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    loss_func_speed = torch.nn.CrossEntropyLoss()

    total_start_time = time.time()

    for epoch in range(args.start_epoch, NUM_EPOCHS):
        # 关键：每个 epoch 开始前设置 sampler 的 epoch，保证数据 shuffle 随机性不同
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        policy.train()

        # 使用 Tensor 累积 Loss 以便多卡 reduce
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_l1 = torch.tensor(0.0, device=device)
        epoch_kl = torch.tensor(0.0, device=device)

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

            n_valid = (~is_pad).sum()
            if n_valid > 0:
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (n_valid * ACTION_DIM + 1e-6)
            else:
                l1 = torch.tensor(0.0, device=device)

            total_kld, _, _ = kl_divergence(mu, logvar)
            kl_loss = total_kld[0]

            speed_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if speed_logits is not None:
                valid_logits = speed_logits[flow_mask]
                valid_labels = speed_labels[flow_mask]
                speed_loss = loss_func_speed(valid_logits, valid_labels)

            loss = l1 + KL_WEIGHT * kl_loss + SPEED_WEIGHT * speed_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积 Loss
            epoch_loss += loss.detach()
            epoch_l1 += l1.detach()
            epoch_kl += kl_loss.detach()

        # --- 6. 聚合所有 GPU 的 Loss (仅用于日志显示) ---
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_l1, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_kl, op=dist.ReduceOp.SUM)

        # 平均 Loss
        avg_loss = epoch_loss.item() / (len(train_loader) * world_size)
        avg_l1 = epoch_l1.item() / (len(train_loader) * world_size)
        avg_kl = epoch_kl.item() / (len(train_loader) * world_size)

        # 仅主进程记录日志
        if rank == 0:
            train_losses.append(avg_loss)
            epoch_dur = time.time() - epoch_start
            eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

            logger.info(
                f"Epoch {epoch:04d} | Loss: {avg_loss:.5f} (L1: {avg_l1:.5f}, KL: {avg_kl:.5f}) | Time: {epoch_dur:.1f}s | ETA: {eta}")

            # 保存模型
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch+1}.ckpt")
                torch.save(policy.state_dict(), save_path)
                save_train_loss_plot(RUN_DIR, train_losses, epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(policy.state_dict(), os.path.join(CKPT_DIR, "policy_best.ckpt"))
                logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    if rank == 0:
        logger.info("Training Done!")

    cleanup()


if __name__ == '__main__':
    """
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/train_maact_2_video_ddp.py --video
    """
    main()