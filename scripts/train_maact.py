import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 引入 MA-ACT 相关模块
from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT
from dataset.efficient_ma_dataset import EfficientEpisodicDataset
from dataset.utils_norm import get_norm_stats


def kl_divergence(mu, logvar):
    """
    计算 KL 散度 Loss (VAE 必要组件)
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def main():
    # =========================================================================
    # 1. 核心配置区域
    # =========================================================================

    # 路径配置
    DATA_DIR = r'F:\projects\lumos\data\20260109'  # 数据集路径
    CKPT_DIR = './checkpoints_maact'  # 模型保存路径
    os.makedirs(CKPT_DIR, exist_ok=True)
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    # YOLO 权重路径 (MA-ACT 计算光流 Mask 必需)
    YOLO_CKPT = r"F:\projects\lumos\ma_act\src\object_detection\object_detection_ckpt\yolov8n.pt"

    # 训练超参数
    NUM_EPOCHS = 5000
    BATCH_SIZE_PER_GPU = 8  # 单卡实际 Batch Size
    TARGET_BATCH_SIZE = 32  # 目标 Batch Size (梯度累积)
    ACCUMULATION_STEPS = max(1, TARGET_BATCH_SIZE // BATCH_SIZE_PER_GPU)

    LR = 1e-4  # 全局(Transformer)学习率
    LR_BACKBONE = 1e-5  # Backbone 专用较小学习率

    CHUNK_SIZE = 100  # 动作预测长度
    KL_WEIGHT = 10.0  # KL Loss 权重系数

    # 机器人与相机配置
    CAMERA_NAMES = ['cam_high']  # 数据集中的相机列表
    MAIN_CAMERA_NAME = 'cam_high'  # 用于计算光流的主相机

    # [关键修正] 维度需匹配您的数据集 (之前报错是因为这里填了14，但数据是7)
    STATE_DIM = 7  # 机械臂状态维度
    ACTION_DIM = 7  # 动作维度

    # MA-ACT 必需历史帧
    N_OBS_STEPS = 2  # 观察历史步数 (>=2)

    print(f"🚀 Training Mode: MA-ACT (SpeedACT)")
    print(f"📦 Batch Size: {BATCH_SIZE_PER_GPU} (Accumulate to {TARGET_BATCH_SIZE})")
    print(f"🔧 LR: {LR}, Backbone LR: {LR_BACKBONE}")
    print(f"📏 Dimensions: State={STATE_DIM}, Action={ACTION_DIM}")

    # =========================================================================
    # 2. 数据集准备
    # =========================================================================

    if not os.path.exists(STATS_PATH):
        print(f"Computing stats from {DATA_DIR}...")
        stats = get_norm_stats(DATA_DIR)
        with open(STATS_PATH, 'wb') as f:
            pickle.dump(stats, f)
    else:
        print(f"Loading stats from {STATS_PATH}...")
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)

    dataset_path_list = glob.glob(os.path.join(DATA_DIR, '*.hdf5'))

    train_dataset = EfficientEpisodicDataset(
        dataset_path_list,
        stats,
        camera_names=CAMERA_NAMES,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=N_OBS_STEPS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2
    )

    # =========================================================================
    # 3. 初始化 MA-ACT 模型
    # =========================================================================
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
        object_detection_ckpt_path=YOLO_CKPT,
        cropped_flow_h=64,
        cropped_flow_w=64,

        feedforward_activation="relu",
        pre_norm=False
    )

    policy = SpeedACT(config).to(device)

    # -----------------------------------------------------------
    # 优化器参数分组 (Backbone 使用低学习率)
    # -----------------------------------------------------------
    param_groups = [
        # 1. Backbone 参数 (LR = 1e-5)
        {
            "params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
        # 2. 其他所有参数 (Transformer, Heads 等) (LR = 1e-4)
        {
            "params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": LR,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # =========================================================================
    # 4. 训练循环
    # =========================================================================
    best_loss = float('inf')
    train_losses = []

    # 定义归一化参数 (ImageNet Stats)
    # 形状: (1, 1, 3, 1, 1) 用于广播匹配 (Batch, Time, Channel, Height, Width)
    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    for epoch in range(NUM_EPOCHS):
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            # 解包数据 (images_list 包含多帧: B, T, C, H, W)
            images_list, qpos, action, is_pad = data

            # 数据移动到 GPU
            # 注意：images_list 里的数据此时是 [0, 1] 范围的 float
            images_list = [x.to(device, non_blocking=True) for x in images_list]
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            # -----------------------------------------------------------
            # 图像归一化 (Normalize to ImageNet Stats)
            # -----------------------------------------------------------
            normalized_images_list = []
            for img in images_list:
                # img shape: (B, T, 3, H, W)
                # 执行广播运算
                norm_img = (img - NORM_MEAN) / NORM_STD
                normalized_images_list.append(norm_img)

            # 构造输入字典
            batch_input = {
                "observation.state": qpos,
                "action": action,
                "action_is_pad": is_pad,
                "observation.images": normalized_images_list  # 使用归一化后的图片
            }
            # 主相机用于光流
            batch_input[MAIN_CAMERA_NAME] = normalized_images_list[0]

            # -----------------------------------------------------------
            # Loss 计算 (L1 + KL)
            # -----------------------------------------------------------

            # 前向传播
            pred_actions, (mu, logvar) = policy(batch_input)

            # L1 Loss (Masked)
            all_l1 = F.l1_loss(pred_actions, action, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            # KL Loss
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            kl_loss = total_kld[0]

            # 总 Loss
            loss = l1 + KL_WEIGHT * kl_loss

            # 梯度累积
            loss_scaled = loss / ACCUMULATION_STEPS
            loss_scaled.backward()

            epoch_loss += loss.item()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理 Epoch 剩余梯度
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        # 日志记录
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.5f} (L1={l1.item():.4f}, KL={kl_loss.item():.4f})")

        # 定期保存与绘图
        if epoch % 500 == 0:
            save_path = os.path.join(CKPT_DIR, f"policy_epoch_{epoch}.ckpt")
            torch.save(policy.state_dict(), save_path)

            # 简单绘图
            plt.figure()
            plt.plot(train_losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(CKPT_DIR, 'loss_curve.png'))
            plt.close()

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CKPT_DIR, "policy_best.ckpt")
            torch.save(policy.state_dict(), save_path)
            print(f"✅ Best model saved with loss {best_loss:.5f}")

    print("Training Done!")


if __name__ == '__main__':
    main()