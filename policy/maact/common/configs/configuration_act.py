from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch


@dataclass
class SpeedACTConfig:
    # --- 核心超参 ---
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    n_obs_steps: int = 1
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- [关键修复] 添加缺失的 Transformer 属性 ---
    feedforward_activation: str = "relu"  # <--- 必须添加这个
    pre_norm: bool = False  # <--- 建议同时添加这个，base_act 经常用到

    # --- 视觉相关 ---
    # 注意：image_features 在 dataclass 初始化时通常为 None，测试时手动注入或由 factory 处理
    image_features: Optional[Dict[str, Tuple[int, int, int]]] = None
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # --- 机器人状态 ---
    # 建议给个默认值 None，避免实例化时报错
    robot_state_feature: Optional[object] = None
    action_feature: Optional[object] = None
    env_state_feature: Optional[object] = None

    # --- VAE 相关 ---
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    # --- SpeedACT 特有配置 ---
    main_camera: str = "camera_front"  # 给个默认值方便测试
    use_optical_flow: bool = True

    global_flow_size: int = 128
    optical_flow_map_height: int = 256
    optical_flow_map_width: int = 320

    # 融合模块参数
    num_speed_categories: int = 3
    speed_loss_weight: float = 0.1
    speed_category_key: str = "speed_category"
    speed_logits_key: str = "speed_logits_internal"
    is_flow_valid_batch_key: str = "is_flow_valid_batch_internal"

    # 预融合 Dropout
    pre_fusion_dropout: float = 0.01