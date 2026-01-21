import os
import sys
import time
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import timedelta
import numpy as np

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# === 引入模块 ===
# 1. 引入之前改好的 Hybrid Dataset
from dataset.efficient_ma_video_dataset import VideoBasedEfficientMADataset
from dataset.utils_norm import get_norm_stats

# 2. 引入模型
from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
# 假设你把 Residual 类放在了这个位置，或者你可以直接把 Residual 类定义贴在这里
from policy.maact.common.model.residual_speed_act import ResidualSpeedACT

# 3. 工具
from scripts.utils_train import setup_logger, get_run_dirs, save_train_loss_plot


def compute_awac_weights(speed_labels, temperature=1.0):
    """
    根据速度标签计算 AWAC 权重。
    假设：速度越快 (Level 2) -> 难度越高 -> 成功后的 Reward 越高 -> 权重越大
    """
    # 简单的映射逻辑：
    # Level 0 (Slow)   -> Advantage = 0.0
    # Level 1 (Normal) -> Advantage = 1.0
    # Level 2 (Fast)   -> Advantage = 2.0
    advantage = speed_labels.float()

    # 计算权重 w = exp(A / T)
    weights = torch.exp(advantage / temperature)

    # 归一化或截断，防止权重过大导致梯度爆炸
    weights = torch.clamp(weights, max=10.0)
    return weights


def main():
    # ==========================
    # 1. 配置区域
    # ==========================
    # 预训练好的 SpeedACT 权重路径 (必须修改！)
    PRETRAINED_CKPT = r'F:\projects\lumos\logs_maact\run_001\checkpoints\policy_best.ckpt'
    DATA_DIR = r'F:\projects\lumos\data\20260109'

    # 训练参数
    BATCH_SIZE = 16
    NUM_EPOCHS = 200  # 残差微调通常很快，不需要太多轮
    LR = 1e-4  # 学习率
    CHUNK_SIZE = 50

    # 混合缓存配置
    MAX_PRELOAD_EPISODES = 50

    # AWAC 温度系数 (越小，对高速数据的偏好越极端)
    AWAC_TEMPERATURE = 1.0

    # 目录设置
    RUN_DIR, CKPT_DIR, RUN_NAME = get_run_dirs("./logs_residual")
    logger = setup_logger(RUN_DIR, name="ResidualRL")
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    logger.info(f"🚀 Residual RL Training Started! Run ID: {RUN_NAME}")

    # ==========================
    # 2. 准备数据
    # ==========================
    # 统计数据
    if not os.path.exists(STATS_PATH):
        stats = get_norm_stats(DATA_DIR)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)
    else:
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)

    # 加载数据集 (使用 Video Dataset)
    import glob
    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))

    train_dataset = VideoBasedEfficientMADataset(
        dataset_path_list, stats, camera_names=['cam_high'],
        chunk_size=CHUNK_SIZE, n_obs_steps=2,  # 注意：Residual 可能会用到多帧
        max_preload_episodes=MAX_PRELOAD_EPISODES
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=4, prefetch_factor=2
    )

    # ==========================
    # 3. 准备模型
    # ==========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A. 初始化配置 (确保与预训练模型一致)
    config = SpeedACTConfig(
        state_dim=8, action_dim=8, chunk_size=CHUNK_SIZE,
        n_obs_steps=2,  # 必须与 Dataset 一致
        # ... 其他参数根据你的实际情况填写 ...
    )

    # B. 加载基座模型
    logger.info(f"Loading base model from {PRETRAINED_CKPT}...")
    base_model = SpeedACT(config)
    state_dict = torch.load(PRETRAINED_CKPT, map_location='cpu')
    base_model.load_state_dict(state_dict)
    base_model.to(device)

    # C. 初始化残差模型
    model = ResidualSpeedACT(base_model, config).to(device)

    # D. 优化器 (关键：只优化 residual_mlp)
    # base_policy 的参数已经在 ResidualSpeedACT.__init__ 里设为 requires_grad=False 了
    # 但为了双重保险，这里显式过滤
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    logger.info(f"Trainable Parameters: {sum(p.numel() for p in trainable_params)}")
    logger.info("Base Policy is FROZEN. 🥶")

    # ==========================
    # 4. 训练循环
    # ==========================
    best_loss = float('inf')
    train_losses = []
    total_start_time = time.time()

    loss_fn = nn.MSELoss(reduction='none')  # 使用 none 以便手动加权

    for epoch in range(NUM_EPOCHS):
        model.train()  # 这里的 train() 只会开启 residual 的 dropout，base 依然是 eval
        epoch_loss = 0
        optimizer.zero_grad()

        epoch_start = time.time()

        for batch_idx, data in enumerate(train_loader):
            # 1. 解包数据 (注意 Dataset 返回了 speed_label)
            image_tensors, qpos, action_gt, is_pad, speed_labels = data

            # 构造 batch 字典适配 model 接口
            batch = {
                "observation.images": image_tensors,  # list of tensors
                "observation.state": qpos.to(device),
                "action": action_gt.to(device),
                "action_is_pad": is_pad.to(device),
                # 用于 SpeedACT 内部逻辑，虽然这里我们不直接用它的 Loss
                "cam_high": image_tensors[0].to(device)
            }

            gt_action = action_gt.to(device)
            speed_labels = speed_labels.to(device)

            # 2. Forward (获取叠加后的动作)
            # pred_action = Base + Residual
            pred_action = model(batch)

            # 3. 计算 AWAC Loss
            # A. 基础 MSE Loss
            # (B, Chunk, Dim)
            raw_loss = loss_fn(pred_action, gt_action)
            # 对 Chunk 和 Dim 维度求平均，保留 Batch 维度 -> (B,)
            mse_per_sample = raw_loss.mean(dim=(1, 2))

            # B. 计算权重 (Importance Sampling / Advantage Weighting)
            # 速度越快 -> 权重越大
            weights = compute_awac_weights(speed_labels, temperature=AWAC_TEMPERATURE)

            # C. 加权最终 Loss
            loss = (mse_per_sample * weights).mean()

            # 4. Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # ==========================
        # 5. 日志与保存
        # ==========================
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        epoch_dur = time.time() - epoch_start
        eta = str(timedelta(seconds=int((NUM_EPOCHS - epoch - 1) * (time.time() - total_start_time) / (epoch + 1))))

        logger.info(f"Epoch {epoch:04d} | AWAC Loss: {avg_loss:.5f} | Time: {epoch_dur:.1f}s | ETA: {eta}")

        if epoch % 50 == 0:  # 残差训练通常存得不需要那么频，或者你可以改频一点
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"residual_epoch_{epoch}.ckpt"))
            save_train_loss_plot(RUN_DIR, train_losses, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 这里保存的是整个 ResidualSpeedACT 的参数（包含 Frozen 的 Base）
            # 实际部署时，你也可以只保存 residual_mlp 的部分，省空间
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "residual_best.ckpt"))
            logger.info(f"⭐ Best Updated: {best_loss:.5f}")

    logger.info("Residual Training Done!")


if __name__ == '__main__':
    main()