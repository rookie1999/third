import torch

from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.speed_act_modulate_full_model import SpeedACT


def test_standalone_model():
    print("🚀 开始初始化配置...")

    # ==========================================
    # 1. 配置 Config
    # ==========================================
    config = SpeedACTConfig(
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=2,  # 测试用，层数设少点
        n_decoder_layers=2,
        chunk_size=50,
        n_obs_steps=2,  # MA-ACT 必须 >= 2
        dropout=0.1,

        # 补全之前报错缺少的属性
        feedforward_activation="relu",
        pre_norm=False,

        # SpeedACT 特有
        main_camera="camera_front",
        use_optical_flow=True,
        optical_flow_map_height=256,
        optical_flow_map_width=320,
        cropped_flow_h=64,
        cropped_flow_w=64,
        num_speed_categories=3,

        # 如果不需要加载实际权重，设为 None
        object_detection_ckpt_path=None,
        pretrained_backbone_weights=None
    )

    # ==========================================
    # 2. [关键修复] 注入 Tensor 而不是 torch.Size
    # ==========================================
    # 模型代码中使用了 config.xxx.shape[0]，所以这里必须传入一个 Tensor
    # 这里的 Tensor 内容不重要，重要的是它的 shape

    # 机器人状态维度 (14,)
    config.robot_state_feature = torch.empty(14)

    # 动作维度 (14,)
    config.action_feature = torch.empty(14)

    # 环境状态 (可选，这里设为 None)
    config.env_state_feature = None

    # 视觉特征：同样需要传入拥有 .shape 属性的 Tensor
    config.image_features = {
        "camera_front": torch.empty(3, 480, 640),
        "camera_wrist": torch.empty(3, 480, 640)
    }

    # ==========================================
    # 3. 实例化模型
    # ==========================================
    print("🏗️ 正在实例化 SpeedACT 模型...")
    try:
        model = SpeedACT(config)
        print("✅ 模型实例化成功！")
    except Exception as e:
        print(f"❌ 模型实例化失败: {e}")
        # 打印详细错误栈以便调试
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 4. 构造虚拟输入数据 (Dummy Batch)
    # ==========================================
    print("📦 构造测试数据...")
    batch_size = 2

    # SpeedACT 需要时序数据: (Batch, Time, ...)
    # Time 维度必须等于 config.n_obs_steps (这里是 2)

    dummy_batch = {
        # 机器人状态: (B, T, D)
        "observation.state": torch.randn(batch_size, config.n_obs_steps, 14),

        # 动作目标: (B, Chunk_Size, D)
        "action": torch.randn(batch_size, config.chunk_size, 14),

        # Mask: 全为 False 表示没有 Padding
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),

        # 图像数据: List 对应 config.image_features 的 keys 顺序
        # 形状: (B, T, C, H, W)
        "observation.images": [
            torch.randn(batch_size, config.n_obs_steps, 3, 480, 640),  # camera_front
            torch.randn(batch_size, config.n_obs_steps, 3, 480, 640)  # camera_wrist
        ]
    }

    # 模拟 LeRobot 数据处理，显式注入主相机数据
    # 因为模型内部会用 batch[config.main_camera] 来获取图像计算光流
    dummy_batch["camera_front"] = dummy_batch["observation.images"][0]

    # ==========================================
    # 5. 前向传播测试
    # ==========================================
    print("▶️ 开始前向传播...")
    model.train()  # 训练模式

    # 如果没有安装光流库或者没有 GPU，这里可能会报错
    # 我们可以尝试 Mock 掉光流部分，或者直接运行看运气
    try:
        # 简单的 Mock 光流编码器 (如果遇到 correlation 报错，请取消下面注释)
        # from unittest.mock import MagicMock
        # model.optical_flow_encoder = MagicMock()
        # model.optical_flow_encoder.return_value = torch.randn(batch_size, 64*64, 512)
        # model.optical_flow_encoder.num_output_tokens = 64*64

        # 注意：如果 perform_yolo_detection 没有正确 Mock 且 ckpt_path 为 None，
        # 代码内部应该会处理返回 None，不会报错。

        actions, (mu, log_sigma) = model(dummy_batch)

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