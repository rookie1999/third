import logging
from itertools import chain

import einops
import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from maact.common.configs.configuration_act import SpeedACTConfig, perform_yolo_detection
from maact.common.model.base_act import ACTSinusoidalPositionEmbedding2d, ACTEncoder, create_sinusoidal_pos_embedding, \
    ACTDecoder
from optical_flow.pwc import predict

# 日志配置：设置为 WARNING 级别，减少不必要的输出
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 在 imports 下方添加这个辅助函数 ---
def get_dim_size(feature, index=0):
    """兼容 LeRobot 的 Feature 对象 (.shape) 和普通 Tuple/List"""
    if hasattr(feature, "shape"):
        return feature.shape[index]
    elif isinstance(feature, (tuple, list)):
        return feature[index]
    else:
        # 假如传进来的既不是对象也不是元组（比如是 None），根据情况处理
        return 0

# +++ NEW +++
# 我们最终设计的、用于替代 EncoderPreFusionModule 的全新融合模块
# 它可以处理不同长度的视觉和光流输入
class DecoupledModulationModule(nn.Module):
    """
    Implements a novel Decoupled Dual-Grained Conditional Modulation for fusion.
    It handles mismatched sequence lengths by using FiLM for global modulation
    and cross-attention for local, fine-grained modulation.
    """

    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads

        # 1. 全局调制器 (Coarse-Grained Global Modulator)
        #    输入是 D_model 维的 aggregated_flow_feature
        self.global_film_generator = nn.Sequential(
            nn.Linear(dim_model, dim_model * 2),
            nn.ReLU(),
            nn.Linear(dim_model * 2, dim_model * 2)
        )

        # 2. 局部调制器 (Fine-Grained Local Modulator via Cross-Attention)
        self.local_cross_attention = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 确保输入输出的 batch 维度在前
        )
        self.dropout_local = nn.Dropout(dropout)
        self.norm_local = nn.LayerNorm(dim_model)

        # 最终输出前的归一化
        self.norm_out = nn.LayerNorm(dim_model)

    def forward(self, visual_tokens: Tensor, flow_tokens: Tensor, is_flow_valid_batch: Tensor) -> Tensor:
        # visual_tokens: (B, N_visual, D)
        # flow_tokens: (B, N_flow, D) -> N_visual 和 N_flow 可以不同
        # is_flow_valid_batch: (B,)

        batch_size = visual_tokens.shape[0]

        # --- 步骤〇：准备工作 ---
        # 对无效样本的 flow_tokens 进行置零，这在注意力中也能起到抑制作用
        flow_gate_mask = is_flow_valid_batch.float().view(batch_size, 1, 1)
        gated_flow_tokens = flow_tokens * flow_gate_mask

        # --- 步骤一：全局定调 (Coarse-Grained Global Tuning) ---
        # 计算全局运动摘要 (只对有效的样本进行)
        aggregated_flow_feature = torch.zeros(batch_size, self.dim_model, device=visual_tokens.device)
        valid_indices = torch.where(is_flow_valid_batch)[0]
        if valid_indices.numel() > 0:
            aggregated_flow_feature[valid_indices] = gated_flow_tokens[valid_indices].mean(dim=1)

        # 生成全局 FiLM 参数
        global_film_params = self.global_film_generator(aggregated_flow_feature)
        gamma_global, beta_global = torch.chunk(global_film_params, 2, dim=-1)

        # 应用全局调制
        globally_modulated_visual = visual_tokens * gamma_global.unsqueeze(1) + beta_global.unsqueeze(1)

        # --- 步骤二：局部精调 (Fine-Grained Local Tuning via Cross-Attention) ---
        # Query: 经过全局调制的视觉token (长度 N_visual)
        # Key, Value: 门控后的光流token (长度 N_flow)
        local_motion_update, _ = self.local_cross_attention(
            query=globally_modulated_visual,
            key=gated_flow_tokens,
            value=gated_flow_tokens
        )  # 输出 local_motion_update 的形状是 (B, N_visual, D)

        # 添加残差连接和归一化，完成局部精调
        fine_tuned_visual = self.norm_local(
            globally_modulated_visual + self.dropout_local(local_motion_update)
        )

        # --- 步骤三：最终输出 ---
        # 此时 fine_tuned_visual 已经是融合后的最终结果
        # 我们再加一个到原始视觉的残差连接，以稳定训练
        final_fused_tokens = self.norm_out(visual_tokens + fine_tuned_visual)

        return final_fused_tokens


class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class OpticalFlowEncoder(nn.Module):
    """Encodes a cropped optical flow map (2 channels: u, v) into a sequence of features."""

    def __init__(self, input_h: int, input_w: int, dim_model: int):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.layer1 = self._make_layer(BasicResidualBlock, 64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicResidualBlock, 64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicResidualBlock, 128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicResidualBlock, 256, 256, num_blocks=2, stride=1)
        self.final_proj = nn.Conv2d(256, dim_model, kernel_size=1)

        dummy_input = torch.zeros(1, 2, input_h, input_w)
        with torch.no_grad():
            dummy_output = self.initial_conv(dummy_input)
            dummy_output = self.layer1(dummy_output)
            dummy_output = self.layer2(dummy_output)
            dummy_output = self.layer3(dummy_output)
            dummy_output = self.layer4(dummy_output)
            dummy_output = self.final_proj(dummy_output)

        self.output_h = dummy_output.shape[-2]
        self.output_w = dummy_output.shape[-1]
        self.num_output_tokens = self.output_h * self.output_w

        self.pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)
        self.norm_out = nn.LayerNorm(dim_model)

        logger.info(
            f"OpticalFlowEncoder output features: H={self.output_h}, W={self.output_w}, Tokens={self.num_output_tokens}")

    def _make_layer(self, block: type[nn.Module], in_channels: int, out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            current_in_channels = in_channels if i == 0 else out_channels
            layers.append(block(current_in_channels, out_channels, s))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_proj(x)

        pos_embed = self.pos_embed(x).to(dtype=x.dtype)
        x = x + pos_embed
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        x = self.norm_out(x)
        return x


class SpeedACT(nn.Module):
    def __init__(self, config: SpeedACTConfig):
        super().__init__()
        assert config.n_obs_steps >= 2, "n_obs_steps must be >= 2 for this version."
        assert config.main_camera != "", "Main camera name cannot be empty"
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature is not None:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    get_dim_size(self.config.robot_state_feature, 0), config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                get_dim_size(self.config.action_feature, 0), config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            self.vae_robot_state_time_embed = nn.Embedding(self.config.n_obs_steps, config.dim_model)
            self.vae_action_time_embed = nn.Embedding(self.config.chunk_size, config.dim_model)

            num_input_token_vae_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature is not None:
                num_input_token_vae_encoder += self.config.n_obs_steps
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_vae_encoder, config.dim_model).unsqueeze(0),
            )

            if self.config.image_features:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

                self.detection_image_transform = T.Resize(
                    size=(self.config.optical_flow_map_height, self.config.optical_flow_map_width),
                    interpolation=T.InterpolationMode.BILINEAR
                )

                self.encoder_img_feat_input_proj = nn.Conv2d(
                    backbone_model.fc.in_features, config.dim_model, kernel_size=1
                )
                self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

                self.visual_feature_downsampler = nn.Sequential(
                    nn.Conv2d(config.dim_model, config.dim_model, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )

                cam_feat = self.config.image_features[self.config.main_camera]
                if hasattr(cam_feat, "shape"):
                    dummy_img_shape_h = cam_feat.shape[-2]
                    dummy_img_shape_w = cam_feat.shape[-1]
                else:
                    # 假设是 tuple (C, H, W)，取倒数第二个和倒数第一个
                    dummy_img_shape_h = cam_feat[-2]
                    dummy_img_shape_w = cam_feat[-1]
                dummy_input_img = torch.zeros(1, 3, dummy_img_shape_h, dummy_img_shape_w, dtype=torch.float32)
                self.backbone.eval()
                with torch.no_grad():
                    dummy_feature_map = self.backbone(dummy_input_img)["feature_map"]
                    dummy_proj_feat = self.encoder_img_feat_input_proj(dummy_feature_map)
                    dummy_downsampled_feat = self.visual_feature_downsampler(dummy_proj_feat)

                self.num_visual_tokens = dummy_downsampled_feat.shape[-2] * \
                                         dummy_downsampled_feat.shape[-1]
                logger.info(
                    f"Visual Encoder output: H={dummy_downsampled_feat.shape[-2]}, W={dummy_downsampled_feat.shape[-1]}, Tokens={self.num_visual_tokens}")

            self.encoder = ACTEncoder(config)
            self.decoder = ACTDecoder(config)

            if self.config.robot_state_feature is not None:
                self.encoder_robot_state_input_proj = nn.Linear(
                    get_dim_size(self.config.robot_state_feature, 0), config.dim_model
                )
            if self.config.env_state_feature:
                self.encoder_env_state_input_proj = nn.Linear(
                    get_dim_size(self.config.env_state_feature, 0), config.dim_model
                )
            self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

            if self.config.object_detection_ckpt_path:
                assert hasattr(config, 'cropped_flow_h') and hasattr(config, 'cropped_flow_w'), \
                    "ACTConfig must define 'cropped_flow_h' and 'cropped_flow_w' for OpticalFlowEncoder."

                self.optical_flow_encoder = OpticalFlowEncoder(
                    input_h=self.config.cropped_flow_h,
                    input_w=self.config.cropped_flow_w,
                    dim_model=self.config.dim_model
                )
                self.num_flow_tokens = self.optical_flow_encoder.num_output_tokens

                self.speed_prediction_head = nn.Sequential(
                    nn.Linear(self.config.dim_model, self.config.dim_model // 2),
                    nn.ReLU(),
                    nn.Linear(self.config.dim_model // 2, self.config.num_speed_categories)
                )

                # speed_token_proj 仍然保留，用于计算辅助损失，但其输出不直接送入主编码器
                self.speed_token_proj = nn.Linear(self.config.num_speed_categories, config.dim_model)

                # === MODIFIED ===
                self.fusion_module = DecoupledModulationModule(
                    dim_model=config.dim_model,
                    num_heads=config.n_heads,
                    dropout=config.pre_fusion_dropout,  # 可以复用这个 dropout 率
                )

            # === MODIFIED ===
            # 重新计算 L_effective (送入 Encoder 的总 token 数量)
            # 基础 token: 潜变量
            self.L_effective = 1
            # 机器人自身状态
            if self.config.robot_state_feature is not None:
                self.L_effective += 1
            # 传感器信息 (视觉+光流融合后，长度与视觉相同)
            if self.config.image_features:
                # self.L_effective += self.num_visual_tokens
                num_cameras = len(self.config.image_features)
                self.L_effective += self.num_visual_tokens * num_cameras
            # 环境状态
            if self.config.env_state_feature:
                self.L_effective += 1

            self.encoder_pos_embed = nn.Embedding(self.L_effective, config.dim_model)
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

            self.action_head = nn.Linear(config.dim_model, get_dim_size(self.config.action_feature, 0))
            self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if hasattr(self, 'speed_prediction_head'):
            for p in self.speed_prediction_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if hasattr(self, 'speed_token_proj'):
            nn.init.xavier_uniform_(self.speed_token_proj.weight)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        # ... (从 batch_size, dtype, device 到 latent_sample 的生成逻辑完全不变) ...
        # ... (从 flow_tokens 的计算到 speed_logits 的生成逻辑也完全不变) ...
        # ... (从 aggregated_visual_tokens 的生成逻辑也完全不变) ...
        # (以上部分很长，直接从您的原始代码复制即可，这里省略以保持清晰)
        if "observation.images" in batch and self.config.main_camera in batch:
            batch_size = batch[self.config.main_camera].shape[0]
        elif "observation.environment_state" in batch:
            batch_size = batch["observation.environment_state"].shape[0]
        elif "observation.state" in batch:
            batch_size = batch["observation.state"].shape[0]
            logger.warning("Batch size inferred from 'observation.state' as no images or env state found.")
        else:
            raise ValueError(
                "Could not determine batch size from input batch. Need images, environment_state, or state.")

        dtype = batch["observation.state"].dtype
        device = batch["observation.state"].device

        mu = log_sigma_x2 = None
        if self.config.use_vae and "action" in batch:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            vae_encoder_input_tokens = [cls_embed]

            if self.config.robot_state_feature is not None:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_time_embed = einops.repeat(self.vae_robot_state_time_embed.weight, "s d -> b s d",
                                                       b=batch_size)
                robot_state_embed = robot_state_embed + robot_state_time_embed
                vae_encoder_input_tokens.append(robot_state_embed)

            action_embed = self.vae_encoder_action_input_proj(batch["action"])
            action_time_embed = einops.repeat(self.vae_action_time_embed.weight, "s d -> b s d", b=batch_size)
            action_embed = action_embed + action_time_embed
            vae_encoder_input_tokens.append(action_embed)

            vae_encoder_input = torch.cat(vae_encoder_input_tokens, axis=1)
            pos_embed_vae_encoder = self.vae_encoder_pos_enc.clone().detach()

            cls_mask = torch.full((batch_size, 1), False, device=device)
            if self.config.robot_state_feature is not None:
                robot_state_mask = torch.full((batch_size, self.config.n_obs_steps), False, device=device)
                vae_encoder_key_padding_mask = torch.cat([cls_mask, robot_state_mask, batch["action_is_pad"]],
                                                         axis=1)
            else:
                vae_encoder_key_padding_mask = torch.cat([cls_mask, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed_vae_encoder.permute(1, 0, 2),
                key_padding_mask=vae_encoder_key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=dtype).to(device)
        flow_tokens = None
        _is_flow_valid_batch_internal = torch.full((batch_size,), False, dtype=torch.bool, device=device)

        if self.config.object_detection_ckpt_path and "observation.images" in batch:
            main_camera_images = batch[self.config.main_camera]
            current_images = main_camera_images[:, -1, :, :, :]
            prev_images = main_camera_images[:, -2, :, :, :]
            resized_prev_images = self.detection_image_transform(prev_images)
            resized_current_images = self.detection_image_transform(current_images)
            current_optical_flow_batch = predict(resized_prev_images, resized_current_images, self.config.device)
            yolo_detection_results = perform_yolo_detection(
                current_images,
                self.config.object_detection_ckpt_path,
                device,
                imgsz=(self.config.optical_flow_map_width, self.config.optical_flow_map_height)
            )
            processed_flow_tokens = torch.zeros(
                batch_size, self.optical_flow_encoder.num_output_tokens, self.config.dim_model,
                dtype=dtype, device=device
            )
            if yolo_detection_results is not None:
                orig_img_h, orig_img_w = current_images.shape[-2:]
                flow_h, flow_w = current_optical_flow_batch.shape[-2:]
                scale_x = flow_w / orig_img_w
                scale_y = flow_h / orig_img_h
                all_boxes_orig_scale = torch.zeros(batch_size, 4, dtype=dtype, device=device)
                has_valid_detection_batch = torch.zeros(batch_size, dtype=torch.bool, device=device)
                valid_boxes_list = []
                valid_indices_list = []
                for b_idx, r in enumerate(yolo_detection_results):
                    if r.boxes and len(r.boxes) > 0:
                        valid_boxes_list.append(r.boxes[0].xyxy[0].to(device))
                        valid_indices_list.append(b_idx)
                if valid_boxes_list:
                    valid_boxes_tensor = torch.stack(valid_boxes_list, dim=0)
                    valid_indices_tensor = torch.tensor(valid_indices_list, dtype=torch.long, device=device)
                    all_boxes_orig_scale[valid_indices_tensor] = valid_boxes_tensor
                    has_valid_detection_batch[valid_indices_tensor] = True
                if has_valid_detection_batch.any():
                    indices_to_process = torch.where(has_valid_detection_batch)[0]
                    temp_boxes = all_boxes_orig_scale[indices_to_process]
                    flow_x1_temp = torch.clamp(torch.floor(temp_boxes[:, 0] * scale_x), 0, flow_w - 1)
                    flow_y1_temp = torch.clamp(torch.floor(temp_boxes[:, 1] * scale_y), 0, flow_h - 1)
                    flow_x2_temp = torch.clamp(torch.ceil(temp_boxes[:, 2] * scale_x), 0, flow_w - 1)
                    flow_y2_temp = torch.clamp(torch.ceil(temp_boxes[:, 3] * scale_y), 0, flow_h - 1)
                    is_valid_dims = (flow_x2_temp > flow_x1_temp) & (flow_y2_temp > flow_y1_temp)
                    final_valid_indices_in_subset = torch.where(is_valid_dims)[0]
                    if final_valid_indices_in_subset.numel() > 0:
                        final_valid_indices_original_batch = indices_to_process[final_valid_indices_in_subset]
                        final_flow_x1 = flow_x1_temp[final_valid_indices_in_subset]
                        final_flow_y1 = flow_y1_temp[final_valid_indices_in_subset]
                        final_flow_x2 = flow_x2_temp[final_valid_indices_in_subset]
                        final_flow_y2 = flow_y2_temp[final_valid_indices_in_subset]
                        x1_norm = 2 * (final_flow_x1 + 0.5) / flow_w - 1
                        y1_norm = 2 * (final_flow_y1 + 0.5) / flow_h - 1
                        x2_norm = 2 * (final_flow_x2 + 0.5) / flow_w - 1
                        y2_norm = 2 * (final_flow_y2 + 0.5) / flow_h - 1
                        base_grid_y, base_grid_x = torch.meshgrid(
                            torch.linspace(-1, 1, self.config.cropped_flow_h, device=device, dtype=dtype),
                            torch.linspace(-1, 1, self.config.cropped_flow_w, device=device, dtype=dtype),
                            indexing='ij'
                        )
                        base_grid = torch.stack((base_grid_x, base_grid_y), dim=-1)
                        grid = torch.empty(
                            (final_valid_indices_original_batch.numel(), self.config.cropped_flow_h,
                             self.config.cropped_flow_w, 2),
                            dtype=dtype, device=device
                        )
                        grid[..., 0] = ((base_grid[..., 0].unsqueeze(0) + 1) / 2) * (x2_norm - x1_norm).unsqueeze(
                            1).unsqueeze(1) + x1_norm.unsqueeze(1).unsqueeze(1)
                        grid[..., 1] = ((base_grid[..., 1].unsqueeze(0) + 1) / 2) * (y2_norm - y1_norm).unsqueeze(
                            1).unsqueeze(1) + y1_norm.unsqueeze(1).unsqueeze(1)
                        sampled_flow_regions = torch.nn.functional.grid_sample(
                            current_optical_flow_batch[final_valid_indices_original_batch],
                            grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                        encoded_flow_regions = self.optical_flow_encoder(sampled_flow_regions)
                        processed_flow_tokens[final_valid_indices_original_batch] = encoded_flow_regions
                        _is_flow_valid_batch_internal[final_valid_indices_original_batch] = True
            flow_tokens = processed_flow_tokens

        speed_logits = torch.zeros(
            batch_size, self.config.num_speed_categories, dtype=dtype, device=device
        )
        if flow_tokens is not None:
            valid_indices = torch.where(_is_flow_valid_batch_internal)[0]
            if valid_indices.numel() > 0:
                valid_flow_tokens = flow_tokens[valid_indices]
                aggregated_valid_flow_feature = valid_flow_tokens.mean(dim=1)
                speed_logits_for_valid = self.speed_prediction_head(aggregated_valid_flow_feature)
                speed_logits[valid_indices] = speed_logits_for_valid
        if self.training:
            batch[self.config.speed_logits_key] = speed_logits
            batch[self.config.is_flow_valid_batch_key] = _is_flow_valid_batch_internal

        # speed_token 仅用于可能的辅助损失计算，不作为主编码器输入
        # speed_token = self.speed_token_proj(speed_logits).unsqueeze(1)

        aggregated_visual_tokens = None
        if self.config.image_features:
            current_cam_images = []
            for cam_id, cam_name in enumerate(self.config.image_features.keys()):
                current_cam_images.append(batch["observation.images"][cam_id][:, -1, :, :, :])
            stacked_images = torch.stack(current_cam_images, dim=1)
            flat_images = einops.rearrange(stacked_images, 'b c img_c h w -> (b c) img_c h w')
            cam_feat = self.backbone(flat_images)["feature_map"]
            cam_feat_proj = self.encoder_img_feat_input_proj(cam_feat)
            cam_feat_proj_downsampled = self.visual_feature_downsampler(cam_feat_proj)
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_feat_proj_downsampled).to(
                dtype=cam_feat_proj_downsampled.dtype)
            processed_tokens = einops.rearrange(cam_feat_proj_downsampled, "n d h w -> n (h w) d") + \
                               einops.rearrange(cam_pos_embed, "n d h w -> n (h w) d")
            aggregated_visual_tokens = processed_tokens.reshape(batch_size, -1, self.config.dim_model)

        # === MODIFIED ===
        # 1. 视觉与光流特征融合
        multimodal_sensor_tokens = None
        if aggregated_visual_tokens is not None:
            if flow_tokens is not None and hasattr(self, 'fusion_module'):
                # 当同时拥有视觉和光流时，调用新的融合模块
                multimodal_sensor_tokens = self.fusion_module(
                    visual_tokens=aggregated_visual_tokens,
                    flow_tokens=flow_tokens,
                    is_flow_valid_batch=_is_flow_valid_batch_internal,
                )
            else:
                multimodal_sensor_tokens = aggregated_visual_tokens

        single_logical_timestep_tokens_list = []

        latent_token = self.encoder_latent_input_proj(latent_sample).unsqueeze(1)
        single_logical_timestep_tokens_list.append(latent_token)

        if self.config.robot_state_feature is not None:
            robot_state_token = self.encoder_robot_state_input_proj(batch["observation.state"][:, -1, :]).unsqueeze(1)
            single_logical_timestep_tokens_list.append(robot_state_token)

        if multimodal_sensor_tokens is not None:
            single_logical_timestep_tokens_list.append(multimodal_sensor_tokens)

        if self.config.env_state_feature:
            env_state_token = self.encoder_env_state_input_proj(
                batch["observation.environment_state"][:, -1, :]).unsqueeze(1)
            single_logical_timestep_tokens_list.append(env_state_token)

        if not single_logical_timestep_tokens_list:
            raise ValueError("Encoder input token list is empty. Check your config and data flow.")

        encoder_in_tokens = torch.cat(single_logical_timestep_tokens_list, dim=1)

        assert encoder_in_tokens.shape[1] == self.L_effective, \
            f"Token sequence length mismatch! Expected {self.L_effective}, but got {encoder_in_tokens.shape[1]}"

        encoder_in_tokens = encoder_in_tokens + self.encoder_pos_embed.weight.unsqueeze(0)
        encoder_in_tokens = encoder_in_tokens.permute(1, 0, 2)

        encoder_out = self.encoder(encoder_in_tokens)

        # ... (Decoder 和 Action Head 部分保持不变) ...
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=dtype,
            device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        actions = self.action_head(decoder_out.transpose(0, 1))

        return actions, (mu, log_sigma_x2)