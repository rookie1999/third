import torch
import time  # 引入 time 模块用于计时

from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT


def test_standalone_model():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始初始化配置... (使用设备: {device})")

    # ==========================================
    # 2. 配置 Config
    # ==========================================
    config = SpeedACTConfig(
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=2,
        n_decoder_layers=2,
        chunk_size=50,
        n_obs_steps=2,
        dropout=0.1,
        feedforward_activation="relu",
        pre_norm=False,
        main_camera="camera_front",
        use_optical_flow=True,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
        num_speed_categories=3,
        pretrained_backbone_weights=None
    )

    # 关键修复：确保 config 内部用于占位的 Tensor 也在正确的设备上（虽然主要是为了取 shape，但保持一致是个好习惯）
    config.robot_state_feature = torch.empty(14).to(device)
    config.action_feature = torch.empty(14).to(device)
    config.env_state_feature = None
    config.image_features = {
        "camera_front": torch.empty(3, 480, 640).to(device),
        "camera_wrist": torch.empty(3, 480, 640).to(device)
    }

    # ==========================================
    # 3. 实例化模型并移动到 GPU
    # ==========================================
    print("🏗️ 正在实例化 SpeedACT 模型...")
    try:
        model = SpeedACT(config)
        model.to(device)  # <--- 关键：将模型移动到 GPU
        print("✅ 模型实例化成功！")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 4. 构造虚拟输入数据 (Dummy Batch) 并移动到 GPU
    # ==========================================
    print("📦 构造测试数据...")
    batch_size = 2

    # 辅助函数：创建数据并移动到 device
    def rand_tensor(*shape):
        return torch.randn(*shape).to(device)

    dummy_batch = {
        # 机器人状态: (B, T, D)
        "observation.state": rand_tensor(batch_size, config.n_obs_steps, 14),

        # 动作目标: (B, Chunk_Size, D)
        "action": rand_tensor(batch_size, config.chunk_size, 14),

        # Mask: (B, Chunk_Size) bool 类型
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool).to(device),

        # 图像数据: List 对应 config.image_features 的 keys 顺序
        # 形状: (B, T, C, H, W)
        "observation.images": [
            rand_tensor(batch_size, config.n_obs_steps, 3, 480, 640),  # camera_front
            rand_tensor(batch_size, config.n_obs_steps, 3, 480, 640)  # camera_wrist
        ]
    }

    # 显式注入主相机数据 (必须也在 device 上)
    dummy_batch["camera_front"] = dummy_batch["observation.images"][0]

    # ==========================================
    # 5. 前向传播测试
    # ==========================================
    print("▶️ 开始前向传播...")
    model.train()

    try:
        for i in range(15):
            start_time = time.time()  # 使用 time.time()
            actions, (mu, log_sigma), _, _ = model(dummy_batch)
            end_time = time.time()
            print(f"⏱️ 推理耗时: {(end_time - start_time):.4f} 秒")
        print("-" * 30)
        print("✅ 前向传播成功！测试通过。")

        print(f"   输出 Actions 形状: {actions.shape} (预期: [{batch_size}, {config.chunk_size}, 14])")

        if mu is not None:
            print(f"   VAE Latent 形状: {mu.shape}")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_standalone_model()