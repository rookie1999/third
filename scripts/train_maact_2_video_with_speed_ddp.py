import argparse
import glob
import os
import pickle
import sys
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist  # [DDP]
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP]
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # [DDP]

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from dataset.efficient_dataset import EfficientEpisodicDataset
from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_with_speed_decoder_query import SpeedACT
from dataset.utils_norm import get_norm_stats
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot, kl_divergence
from dataset.efficient_ma_dynamic_video_dataset import VideoBasedEfficientMADataset


# [DDP] 工具函数：检查是否为主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser(description="ACT DDP Training Script")
    parser.add_argument('--video', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--fisheye', action='store_true', help='Use video dataset (load from .mp4)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start from')
    args = parser.parse_args()

    # [DDP] 1. 初始化分布式环境
    # torchrun 会自动设置 LOCAL_RANK 等环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f"cuda:{local_rank}")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    TARGET_SIZE = (640, 480)
    if args.fisheye:
        TARGET_SIZE = (480, 480)

    # 路径配置
    if args.video:
        DATA_DIR = r'/root/Users/zhanguozhi/lumos/data/012_rot/episode'
    else:
        DATA_DIR = r'F:\projects\lumos\data\20260109'  # 注意：Linux下路径格式不同，确保多卡环境是Linux

    # [DDP] 仅在主进程设置 Logger 和创建目录
    logger = None
    CKPT_DIR = None
    RUN_DIR = None

    if is_main_process():
        RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_maact")
        logger = setup_logger(RUN_DIR, name="MA_ACT")
        STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

        mode_str = "VIDEO-Based" if args.video else "HDF5-RAM-Based"
        logger.info(f"🚀 MA-ACT DDP Training Started! Mode: [{mode_str}] | Run ID: {RUN_NAME}")
        logger.info(f"Using {world_size} GPUs.")
    else:
        # 其他进程只需知道 STATS_PATH 在哪里（通常需要指向一个公共路径，或者由 Rank 0 广播）
        # 这里为了简单，假设 stats 路径逻辑需要同步。
        # 更好的做法是：Rank 0 计算完 stats 后，广播给其他进程，或者存到一个固定位置。
        # 下面逻辑中我们会用 barrier 解决 stats 计算冲突。
        # 注意：这里 CKPT_DIR 对非 Rank0 是 None，后续要小心使用
        pass

    # 超参数配置
    NUM_EPOCHS = 2000
    BATCH_SIZE = 64
    LR = 1e-4
    LR_BACKBONE = 1e-5
    CHUNK_SIZE = 50
    KL_WEIGHT = 10.0
    CLS_WEIGHT = 0.2

    CAMERA_NAMES = ['cam_high']
    MAIN_CAMERA_NAME = 'cam_high'
    N_OBS_STEPS = 1
    NUM_SPEED_CATEGORIES = 3

    num_workers = 7

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))

    stats = None
    if is_main_process():
        # Rank 0 负责计算或加载
        # 注意：为了让其他进程也能读取，STATS_PATH 最好是一个公共可读路径，而不是动态生成的 CKPT_DIR
        # 这里我们临时生成，然后通过 torch.save/load 或者广播同步
        logger.info(f"Rank 0: Preparing stats from {DATA_DIR}...")
        stats = get_norm_stats(DATA_DIR)
        # 如果需要保存到磁盘供后续使用
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)

    # [DDP] 同步 Stats 数据
    # 将 stats 对象广播给所有进程
    stats_list = [stats]
    dist.broadcast_object_list(stats_list, src=0)
    stats = stats_list[0]

    # 等待同步完成
    dist.barrier()

    STATE_DIM = stats['qpos_mean'].shape[0]
    ACTION_DIM = stats['action_mean'].shape[0]

    # [DDP] 数据集初始化 (所有进程都需要初始化数据集)
    if args.video:
        if is_main_process(): logger.info(f"Initializing Video MA-Dataset...")
        train_dataset = VideoBasedEfficientMADataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS,
            target_size=TARGET_SIZE
        )
    else:
        if is_main_process(): logger.info(f"Initializing HDF5 MA-Dataset...")
        train_dataset = EfficientEpisodicDataset(
            dataset_path_list, stats,
            camera_names=CAMERA_NAMES,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=N_OBS_STEPS
        )

    # [DDP] 2. 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # [DDP] 必须设为 False，shuffle 由 sampler 控制
        sampler=train_sampler,  # [DDP] 传入 sampler
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2
    )

    # 模型初始化
    config = SpeedACTConfig(
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS,
        image_features={cam: (3, TARGET_SIZE[1], TARGET_SIZE[0]) for cam in CAMERA_NAMES},
        main_camera=MAIN_CAMERA_NAME,
        robot_state_feature=(STATE_DIM,),
        action_feature=(ACTION_DIM,),
        use_optical_flow=False,
        num_speed_categories=NUM_SPEED_CATEGORIES,
        feedforward_activation="relu",
        pre_norm=False,
        global_flow_size=128,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
    )

    policy = SpeedACT(config)
    policy.to(device)  # 先移动到对应 GPU

    # [DDP] 3. SyncBatchNorm (可选，建议开启) 和 DDP 封装
    policy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(policy)
    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank)

    # 优化器需要处理的是 policy.parameters() (此时已经是 DDP 包装后的)
    # DDP 包装后，访问原始参数名通常会有 "module." 前缀，但 named_parameters 会自动处理
    param_groups = [
        {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process(): logger.info(f"🔄 Resuming training from checkpoint: {args.resume}")
            # map_location 必须指定，否则会全部加载到 GPU 0
            checkpoint = torch.load(args.resume, map_location=device)
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            # 注意：必须使用 policy.module.load_state_dict 而不是 policy.load_state_dict
            policy.module.load_state_dict(new_state_dict)
            if is_main_process(): logger.info(f"✅ Loaded weights successfully.")
        else:
            if is_main_process(): logger.error(f"❌ Checkpoint file not found: {args.resume}")
            return

    best_loss = float('inf')
    train_losses = []

    if N_OBS_STEPS == 1:
        NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    else:
        NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    total_start_time = time.time()

    for epoch in range(args.start_epoch, NUM_EPOCHS):
        # [DDP] 4. 每个 Epoch 开始时必须调用，以确保数据打乱的随机种子不同
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        policy.train()
        epoch_loss, epoch_l1, epoch_kl = 0, 0, 0
        optimizer.zero_grad()

        # 进度条建议仅在主进程显示，或者静默处理
        # 这里直接遍历
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
                "observation.images": norm_imgs, "speed_label": speed_labels
            }

            pred_actions, (mu, logvar), pred_speed_logits = policy(batch_input)

            all_l1 = F.l1_loss(pred_actions, action, reduction='none')

            n_valid = (~is_pad).sum()
            if n_valid > 0:
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (n_valid * ACTION_DIM + 1e-6)
            else:
                l1 = torch.tensor(0.0, device=device)

            total_kld, _, _ = kl_divergence(mu, logvar)
            kl_loss = total_kld[0]

            loss = l1 + KL_WEIGHT * kl_loss

            if pred_speed_logits is not None:
                loss_cls = F.cross_entropy(pred_speed_logits, speed_labels)
                loss += CLS_WEIGHT * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # [DDP] Loss 记录建议：
            # 严格来说应该用 dist.all_reduce 平均所有卡的 loss 才能得到准确的全局 loss
            # 但为了性能和简单，通常只打印 Rank 0 的 loss 作为参考即可
            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_kl += kl_loss.item()

        # 计算平均 Loss (这里仅反映当前 GPU 的 Loss)
        avg_loss = epoch_loss / len(train_loader)

        # [DDP] 可选：同步所有卡的 Loss 均值以便保存模型判断
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss_global = avg_loss_tensor.item()

        # 仅在 Rank 0 进行记录和保存
        if is_main_process():
            train_losses.append(avg_loss_global)
            epoch_dur = time.time() - epoch_start
            eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

            logger.info(
                f"Epoch {epoch:04d} | Global Loss: {avg_loss_global:.5f} | Time: {epoch_dur:.1f}s | ETA: {eta}")

            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch}.ckpt")
                # 保存时建议保存 policy.module (去掉 DDP 包装)，方便单卡推理加载
                torch.save(policy.module.state_dict(), save_path)
                save_train_loss_plot(RUN_DIR, train_losses, epoch)

            if avg_loss_global < best_loss:
                best_loss = avg_loss_global
                torch.save(policy.module.state_dict(), os.path.join(CKPT_DIR, "policy_best.ckpt"))
                logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    if is_main_process():
        logger.info("Training Done!")

    # [DDP] 销毁进程组
    dist.destroy_process_group()


if __name__ == '__main__':
    """
    torchrun --nproc_per_node=4 scripts/train_maact_2_video_with_speed_ddp.py --video
    CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7  torchrun --nproc_per_node=7 scripts/train_maact_2_video_with_speed_ddp.py --video --resume /root/Users/zhanguozhi/projects/replay_remote_ctrl/logs_maact/policy_epoch_699.ckpt --start_epoch=700
    """
    main()