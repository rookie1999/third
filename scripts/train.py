import os
import sys

import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import glob
import pickle
import matplotlib.pyplot as plt

from dataset.efficient_dataset import EfficientEpisodicDataset
from dataset.utils_norm import get_norm_stats
from policy.act.policy import ACTPolicy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def main():
    # === 2. 配置参数 (针对新机器人调整) ===
    # 修改路径指向新数据的文件夹
    DATA_DIR = r'F:\projects\lumos\data\20260109'
    CKPT_DIR = './checkpoints'

    # --- 梯度累积策略 (显存优化) ---
    # 你的显存一次只能跑 8 个，但为了训练效果，我们模拟 32 个
    BATCH_SIZE_PER_GPU = 8
    TARGET_BATCH_SIZE = 32
    ACCUMULATION_STEPS = TARGET_BATCH_SIZE // BATCH_SIZE_PER_GPU

    NUM_EPOCHS = 100000
    # 针对新数据，学习率可以保持 1e-5，如果收敛太慢可尝试 2e-5
    LR = 1e-5
    CHUNK_SIZE = 50

    # 【核心修改】根据 HDF5 info，qpos 和 action 都是 7 维
    STATE_DIM = 7

    CAMERA_NAMES = ['cam_high']

    # Windows 下 Worker 设置
    # 如果遇到 h5py 报错或死锁，请将此改为 0
    NUM_WORKERS = 4

    if not os.path.exists(CKPT_DIR): os.makedirs(CKPT_DIR)

    # === 3. 准备数据 ===
    stats = get_norm_stats(DATA_DIR)
    with open(os.path.join(CKPT_DIR, 'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    dataset_files = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))

    # 策略：先全量训练试试 (根据你的上一条需求)
    USE_ALL_DATA_FOR_TRAIN = True

    if USE_ALL_DATA_FOR_TRAIN:
        print(f"【注意】发现 {len(dataset_files)} 个文件，使用全量数据训练！")
        train_files = dataset_files
        val_files = dataset_files  # 占位，不使用
    else:
        split_idx = int(len(dataset_files) * 0.9)
        if split_idx == 0: split_idx = 1
        train_files = dataset_files[:split_idx]
        val_files = dataset_files[split_idx:]

    # 使用高效 Dataset
    train_dataset = EfficientEpisodicDataset(train_files, stats, CAMERA_NAMES, CHUNK_SIZE, use_cache=True)
    val_dataset = EfficientEpisodicDataset(val_files, stats, CAMERA_NAMES, CHUNK_SIZE, use_cache=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )

    # === 4. 构建模型 ===
    policy_config = {
        'lr': LR,
        'num_queries': CHUNK_SIZE,
        'kl_weight': 10,
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        # 【重要】确保 backbone.py 已经修改为加载 ImageNet 权重
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': CAMERA_NAMES,
        'state_dim': STATE_DIM  # 传入 7
    }

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()

    # === 5. 训练循环 (含梯度累积) ===
    train_losses = []

    for epoch in range(NUM_EPOCHS):
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            image, qpos, action, is_pad = [x.cuda(non_blocking=True) for x in data]

            # 1. 前向计算
            loss_dict = policy(qpos, image, actions=action, is_pad=is_pad)

            # 2. Loss 缩放 (为了梯度累积)
            loss = loss_dict['loss'] / ACCUMULATION_STEPS

            # 3. 反向传播 (此时梯度累加，不更新参数)
            loss.backward()

            # 4. 只有当累计够了步数，才更新参数
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 记录 Loss (乘回来方便观察)
            epoch_loss += loss.item() * ACCUMULATION_STEPS

        # 处理 Epoch 最后剩余的梯度
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.5f}")

        if epoch % 100 == 0:
            # 保存
            torch.save(policy.state_dict(), os.path.join(CKPT_DIR, f'policy_epoch_{epoch}.ckpt'))
            plt.plot(train_losses)
            plt.savefig(os.path.join(CKPT_DIR, 'loss_curve.png'))
            plt.close()

    torch.save(policy.state_dict(), os.path.join(CKPT_DIR, 'policy_last.ckpt'))
    print("Training Done!")


if __name__ == "__main__":
    main()