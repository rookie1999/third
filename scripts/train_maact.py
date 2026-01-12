import os
import sys
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from maact.common.configs.configuration_act import SpeedACTConfig
from maact.common.model.speed_act_modulate_full_model import SpeedACT
from dataset.efficient_ma_dataset import EfficientEpisodicDataset
from dataset.utils_norm import get_norm_stats

# 导入两种策略
# from policy.act.policy import ACTPolicy  # 旧版 Standard ACT


def main():
    # =========================================================================
    # 1. 核心配置区域 (修改这里来适配你的训练)
    # =========================================================================

    # [关键开关] True = 训练 MA-ACT (SpeedACT); False = 训练普通 ACT
    USE_SPEED_ACT = True

    # 数据与保存路径
    DATA_DIR = r'F:\projects\lumos\data\20260109'  # 你的数据路径
    CKPT_DIR = './checkpoints'  # 模型保存路径
    os.makedirs(CKPT_DIR, exist_ok=True)
    STATS_PATH = os.path.join(CKPT_DIR, 'dataset_stats.pkl')

    # 训练超参数
    NUM_EPOCHS = 5000
    BATCH_SIZE_PER_GPU = 8  # 实际单次前向的 Batch Size (受显存限制)
    TARGET_BATCH_SIZE = 32  # 目标 Batch Size (通过梯度累积实现)
    ACCUMULATION_STEPS = max(1, TARGET_BATCH_SIZE // BATCH_SIZE_PER_GPU)

    LR = 1e-4
    CHUNK_SIZE = 100  # 预测未来多少步

    # 机器人与相机配置
    CAMERA_NAMES = ['cam_high']  # 你的数据集中的相机列表
    MAIN_CAMERA_NAME = 'cam_high'  # MA-ACT 需要指定主相机计算光流

    # 状态维度配置
    STATE_DIM = 14  # 机械臂状态维度 (例如 7关节 + 7速度)
    ACTION_DIM = 14  # 动作维度

    # YOLO 权重路径 (仅 MA-ACT 需要)
    YOLO_CKPT = r"F:\projects\lumos\ma_act\src\object_detection\object_detection_ckpt\yolov8n.pt"

    print(f"🚀 Training Mode: {'MA-ACT (SpeedACT)' if USE_SPEED_ACT else 'Standard ACT'}")
    print(f"📦 Batch Size: {BATCH_SIZE_PER_GPU} (Accumulate to {TARGET_BATCH_SIZE})")

    # =========================================================================
    # 2. 初始化 Dataset 和 DataLoader
    # =========================================================================

    # 自动计算统计数据 (Mean/Std)
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

    # [关键] 根据模式设定 n_obs_steps
    # MA-ACT 需要至少 2 帧来计算光流；ACT 只需要 1 帧
    current_n_obs_steps = 2 if USE_SPEED_ACT else 1

    train_dataset = EfficientEpisodicDataset(
        dataset_path_list,
        stats,
        camera_names=CAMERA_NAMES,
        chunk_size=CHUNK_SIZE,
        n_obs_steps=current_n_obs_steps
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
    # 3. 初始化模型 (Policy)
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if USE_SPEED_ACT:
        # --- 初始化 MA-ACT ---
        config = SpeedACTConfig(
            dim_model=512,
            n_heads=8,
            dim_feedforward=3200,
            n_encoder_layers=4,
            n_decoder_layers=1,
            chunk_size=CHUNK_SIZE,
            n_obs_steps=current_n_obs_steps,  # 必须 >= 2

            # 视觉配置
            image_features={cam: (3, 480, 640) for cam in CAMERA_NAMES},  # 假设图片都是 480x640
            main_camera=MAIN_CAMERA_NAME,

            # 状态配置
            robot_state_feature=(STATE_DIM,),  # 注意这是 tuple
            action_feature=(ACTION_DIM,),

            # 功能开关
            use_optical_flow=True,
            object_detection_ckpt_path=YOLO_CKPT,

            # 光流参数
            cropped_flow_h=64,
            cropped_flow_w=64,

            # 缺失属性补全 (防止报错)
            feedforward_activation="relu",
            pre_norm=False
        )
        policy = SpeedACT(config).to(device)
    # else:
    #     # --- 初始化 Standard ACT ---
    #     policy = ACTPolicy(
    #         action_dim=ACTION_DIM,
    #         state_dim=STATE_DIM,
    #         hidden_dim=512,
    #         chunk_size=CHUNK_SIZE,
    #         # 如果你有特定的 VAE 或 Backbone 参数，请在这里添加
    #     ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)

    # =========================================================================
    # 4. 训练循环
    # =========================================================================
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        policy.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            # 解包数据 (来自 efficient_dataset.py 的 __getitem__)
            # images_list: 如果 n_obs=1 是 [(B,C,H,W)...], 如果 n_obs=2 是 [(B,T,C,H,W)...]
            images_list, qpos, action, is_pad = data

            # 数据移动到 GPU
            images_list = [x.to(device, non_blocking=True) for x in images_list]
            qpos = qpos.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            is_pad = is_pad.to(device, non_blocking=True)

            # --- 分支：数据输入模型 ---
            if USE_SPEED_ACT:
                # [MA-ACT 分支] 构造字典输入
                batch_input = {
                    "observation.state": qpos,  # (B, T, D)
                    "action": action,  # (B, Chunk, D)
                    "action_is_pad": is_pad,  # (B, Chunk)
                    "observation.images": images_list  # List[(B, T, C, H, W)]
                }
                # 手动注入主相机数据用于光流计算
                # 假设 config.main_camera 对应的就是 images_list[0] (如果是单摄)
                # 如果是多摄，请根据 camera_names 的顺序索引，这里默认取第一个
                batch_input[MAIN_CAMERA_NAME] = images_list[0]

                loss_dict = policy(batch_input)

            else:
                # [Standard ACT 分支] 参数列表输入
                # ACT 通常只接受单张图片（或者多张 concat）
                # 这里假设取第一个相机的图像
                image_input = images_list[0]  # (B, C, H, W)

                loss_dict = policy(qpos, image_input, actions=action, is_pad=is_pad)

            # --- Loss 处理与反向传播 ---
            loss = loss_dict['loss']

            # 梯度累积：Loss 除以步数
            loss_scaled = loss / ACCUMULATION_STEPS
            loss_scaled.backward()

            # 记录真实 Loss
            epoch_loss += loss.item()

            # 执行更新
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理 Epoch 结尾剩余的梯度
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        # 打印日志
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.5f}")

        # 保存权重
        if epoch % 500 == 0:
            ckpt_name = f"policy_epoch_{epoch}_ma_act.ckpt" if USE_SPEED_ACT else f"policy_epoch_{epoch}_act.ckpt"
            save_path = os.path.join(CKPT_DIR, ckpt_name)
            torch.save(policy.state_dict(), save_path)

        # 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_name = "policy_best_ma_act.ckpt" if USE_SPEED_ACT else "policy_best_act.ckpt"
            save_path = os.path.join(CKPT_DIR, ckpt_name)
            torch.save(policy.state_dict(), save_path)
            print(f"✅ Best model saved with loss {best_loss:.5f}")


if __name__ == '__main__':
    main()