import argparse
import glob
import os
import pickle
import sys
import time
import json
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist  # [DDP] 引入分布式模块
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [DDP] 引入分布式采样器

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from policy.act.policy import ACTPolicy
from dataset.utils_norm import get_norm_stats
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot

from dataset.efficient_dataset import EfficientEpisodicDataset
from dataset.efficient_video_dataset import VideoBasedEfficientDataset


# [DDP] 工具函数：判断是否为主进程 (Rank 0)
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser(description="ACT DDP Training Script")
    parser.add_argument('--video', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--fisheye', action='store_true', help='Whether use fisheye camera or not')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from')
    args = parser.parse_args()

    # [DDP] 1. 初始化分布式环境
    # torchrun 启动时会自动设置 LOCAL_RANK 环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f"cuda:{local_rank}")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 路径配置
    if args.video:
        DATA_DIR = r'/home/zgz/projects/lumos/train_data/episode'  # 确保多卡机器路径正确
    else:
        DATA_DIR = r'/home/zgz/projects/lumos/data/20260109'  # Linux 路径

    # [DDP] 仅在主进程中设置 Logger 和创建文件夹
    logger = None
    CKPT_DIR = None
    RUN_DIR = None
    STATS_PATH = None
    RUN_NAME = None

    if is_main_process():
        RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_act")
        STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')
        logger = setup_logger(RUN_DIR, name="ACT")

        mode_str = "VIDEO-Based" if args.video else "HDF5-RAM-Based"
        logger.info(f"🚀 ACT DDP Training Started! Mode: [{mode_str}] | Run ID: {RUN_NAME}")
        logger.info(f"Using {world_size} GPUs. Data Dir: {DATA_DIR}")
    else:
        # 非主进程不打印 Log，或者可以设置一个 Dummy Logger
        pass

    # 超参数
    NUM_EPOCHS = 1000
    # [DDP] Batch Size 这里指每张卡的 batch size
    # 如果原来总 Batch Size 是 64，现在有 4 张卡，这里设为 16 即可保持总数不变；或者设为 64 加速训练
    BATCH_SIZE = 64
    LR = 1e-5
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    CAMERA_NAMES = ['cam_high']

    num_workers = 6

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))

    # [DDP] 统计数据 (Stats) 同步
    # 策略：Rank 0 计算 -> 广播给其他 Rank
    stats = None
    if is_main_process():
        if len(dataset_path_list) == 0:
            logger.error(f"No HDF5 files found in {DATA_DIR}. Check your path!")
            sys.exit(1)

        logger.info(f"Computing/Loading stats from: {DATA_DIR} ...")
        # 实际项目中，建议 stats 预先算好存固定位置，避免每次计算
        # 这里为了兼容，依然实时计算或从本次 Run 目录加载
        stats = get_norm_stats(DATA_DIR)

        # 保存一份到本次 Log 目录备份
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)

    TARGET_SIZE = (640, 480)
    if args.fisheye:
        TARGET_SIZE = (480, 480)

    # 广播 stats 对象
    stats_list = [stats]
    dist.broadcast_object_list(stats_list, src=0)
    stats = stats_list[0]

    # 等待同步
    dist.barrier()

    STATE_DIM = stats['qpos_mean'].shape[0]
    ACTION_DIM = stats['action_mean'].shape[0]
    if STATE_DIM == 7:
        logger.info("Use rpy for training")
    elif STATE_DIM == 10:
        logger.info("Use rot6d for training")


    if args.video:
        if is_main_process(): logger.info("Initializing VideoBasedEfficientDataset...")
        train_dataset = VideoBasedEfficientDataset(
            dataset_path_list, stats, camera_names=CAMERA_NAMES, chunk_size=CHUNK_SIZE,
            target_size=TARGET_SIZE
        )
    else:
        if is_main_process(): logger.info("Initializing EfficientEpisodicDataset...")
        train_dataset = EfficientEpisodicDataset(
            dataset_path_list, stats, camera_names=CAMERA_NAMES, chunk_size=CHUNK_SIZE
        )

    # [DDP] 2. 使用 DistributedSampler
    # shuffle=True 表示每个 Epoch 数据会打乱
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # [DDP] 必须为 False，shuffle 交给 sampler
        sampler=train_sampler,  # [DDP] 传入 sampler
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

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

    # 仅主进程打印配置
    if is_main_process():
        config_log = {
            "Experiment Info": {
                "Run Name": RUN_NAME,
                "Mode": mode_str,
                "Device": str(device),
                "Data Dir": DATA_DIR,
                "Resume Path": args.resume if args.resume else "None"
            },
            "Training Hyperparams": {
                "Num Epochs": NUM_EPOCHS,
                "Batch Size (Per GPU)": BATCH_SIZE,
                "Global Batch Size": BATCH_SIZE * world_size,
                "Learning Rate": LR,
                "Chunk Size": CHUNK_SIZE,
                "Camera Names": CAMERA_NAMES
            },
            "Model Architecture": args_override
        }
        logger.info("-" * 60)
        logger.info("🔧 HYPERPARAMETERS CONFIGURATION:")
        logger.info("\n" + json.dumps(config_log, indent=4, default=str))
        logger.info("-" * 60)

    # 模型初始化
    policy = ACTPolicy(args_override)
    policy.to(device)  # 先移至 GPU

    # [DDP] 3. SyncBatchNorm (推荐) 和 DDP 封装
    # 如果模型中有 BatchNorm 层，这步能同步均值方差，提升多卡训练效果
    policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy)

    # find_unused_parameters=False 通常能提升速度，除非模型有些层在前向传播中未被使用
    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 获取 optimizer (注意：DDP 包装后，configure_optimizers 可能会失效，因为那是原模型的方法)
    # policy 现在是 DDP 对象，policy.module 才是原来的 ACTPolicy
    optimizer = policy.module.configure_optimizers()

    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process(): logger.info(f"🔄 Resuming training from checkpoint: {args.resume}")
            # map_location 必须指定
            checkpoint = torch.load(args.resume, map_location=device)
            policy.module.load_state_dict(checkpoint)  # 加载到 module
            if is_main_process(): logger.info(f"✅ Loaded weights successfully.")
        else:
            if is_main_process(): logger.error(f"❌ Checkpoint file not found: {args.resume}")
            # 非主进程也要退出
            dist.destroy_process_group()
            return

    best_loss = float('inf')
    train_losses = []
    total_start_time = time.time()

    for epoch in range(args.start_epoch, NUM_EPOCHS):
        # [DDP] 4. 设置 Sampler 的 epoch，保证每个 epoch 数据乱序不同
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            image_tensor, qpos, action, is_pad = data

            image = image_tensor.to(device, non_blocking=True)
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            # 调用 DDP 模型
            loss_dict = policy(qpos, image, action, is_pad)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # 计算当前 GPU 的平均 Loss
        avg_loss = epoch_loss / len(train_loader)

        # [DDP] 聚合所有 GPU 的 Loss 以便记录和保存 Best Model
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        global_avg_loss = avg_loss_tensor.item()

        # 仅主进程负责记录和保存
        if is_main_process():
            train_losses.append(global_avg_loss)
            epoch_dur = time.time() - epoch_start
            eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

            logger.info(f"Epoch {epoch:04d} | Global Loss: {global_avg_loss:.5f} | Time: {epoch_dur:.1f}s | ETA: {eta}")

            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch}.ckpt")
                # 保存 policy.module.state_dict()，去掉 DDP 的 module 前缀
                torch.save(policy.module.state_dict(), save_path)
                save_train_loss_plot(RUN_DIR, train_losses, epoch)

            if global_avg_loss < best_loss:
                best_loss = global_avg_loss
                torch.save(policy.module.state_dict(), os.path.join(CKPT_DIR, "policy_best.ckpt"))
                logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    if is_main_process():
        logger.info("Training Done!")

    # 销毁进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    """
    torchrun --nproc_per_node=4 train_act_ddp.py --video
    """
    main()