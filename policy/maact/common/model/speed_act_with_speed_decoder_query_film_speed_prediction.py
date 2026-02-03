import logging
from itertools import chain

import einops
import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

from policy.maact.common.configs.configuration_act import SpeedACTConfig
from policy.maact.common.model.base_act import ACTSinusoidalPositionEmbedding2d, ACTEncoder, \
    create_sinusoidal_pos_embedding, ACTDecoder

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_dim_size(feature, index=0):
    """兼容 LeRobot 的 Feature 对象 (.shape) 和普通 Tuple/List"""
    if hasattr(feature, "shape"):
        return feature.shape[index]
    elif isinstance(feature, (tuple, list)):
        return feature[index]
    else:
        # 假如传进来的既不是对象也不是元组（比如是 None），根据情况处理
        return 0


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


class SpeedACT(nn.Module):
    def __init__(self, config: SpeedACTConfig):
        super().__init__()
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
            self.vae_action_time_embed = nn.Embedding(self.config.chunk_size, config.dim_model)

            num_input_token_vae_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature is not None:
                num_input_token_vae_encoder += self.config.n_obs_steps
            num_input_token_vae_encoder += 1  # 速度token
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

            self.speed_embedding = nn.Embedding(self.config.num_speed_categories, config.dim_model)

            self.L_effective = 1
            if self.config.robot_state_feature is not None:
                self.L_effective += 1
            if self.config.image_features:
                num_cameras = len(self.config.image_features)
                self.L_effective += self.num_visual_tokens * num_cameras
            if self.config.env_state_feature:
                self.L_effective += 1

            self.encoder_pos_embed = nn.Embedding(self.L_effective, config.dim_model)
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

            self.action_head = nn.Linear(config.dim_model, get_dim_size(self.config.action_feature, 0))
            self._reset_parameters()

        self.speed_film_gen = nn.Linear(config.dim_model, config.dim_model * 2)
        # film参数的零初始化，这能防止训练初期和后期的数值爆炸，保证初始状态下图像特征不被扭曲
        nn.init.constant_(self.speed_film_gen.weight, 0)
        nn.init.constant_(self.speed_film_gen.bias, 0)
        self.null_speed_embed = nn.Parameter(torch.randn(1, config.dim_model) * 0.02)
        self.speed_drop_prob = 0.25

        self.speed_prediction_head = nn.Linear(config.dim_model, config.num_speed_categories)

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

        nn.init.xavier_uniform_(self.speed_prediction_head.weight)
        nn.init.constant_(self.speed_prediction_head.bias, 0)

    def forward(self, batch: dict[str, Tensor], return_features: bool = False) -> tuple[
        Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if "observation.images" in batch and self.config.main_camera in batch:
            batch_size = batch[self.config.main_camera].shape[0]
        elif "observation.environment_state" in batch:
            batch_size = batch["observation.environment_state"].shape[0]
        elif "observation.state" in batch:
            batch_size = batch["observation.state"].shape[0]
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
                if robot_state_embed.dim() == 2:
                    robot_state_embed = robot_state_embed.unsqueeze(1)
                vae_encoder_input_tokens.append(robot_state_embed)

            speed_idx_vae = batch["speed_label"]  # (B,)
            speed_token_vae = self.speed_embedding(speed_idx_vae).unsqueeze(1)  # (B, 1, dim_model)

            vae_encoder_input_tokens.append(speed_token_vae)

            action_embed = self.vae_encoder_action_input_proj(batch["action"])
            action_time_embed = einops.repeat(self.vae_action_time_embed.weight, "s d -> b s d", b=batch_size)
            action_embed = action_embed + action_time_embed
            vae_encoder_input_tokens.append(action_embed)

            vae_encoder_input = torch.cat(vae_encoder_input_tokens, axis=1)
            pos_embed_vae_encoder = self.vae_encoder_pos_enc.clone().detach()

            cls_mask = torch.full((batch_size, 1), False, device=device)
            speed_mask = torch.full((batch_size, 1), False, device=device)
            if self.config.robot_state_feature is not None:
                robot_state_mask = torch.full((batch_size, self.config.n_obs_steps), False, device=device)
                vae_encoder_key_padding_mask = torch.cat(
                    [cls_mask, robot_state_mask, speed_mask, batch["action_is_pad"]],
                    axis=1)
            else:
                vae_encoder_key_padding_mask = torch.cat(
                    [cls_mask, speed_mask, batch["action_is_pad"]], axis=1)

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

        speed_idx = batch["speed_label"]
        current_speed_embed = self.speed_embedding(speed_idx)  # (B, D)
        if self.training and self.speed_drop_prob > 0:
            mask = torch.rand(batch_size, 1, device=device) < self.speed_drop_prob
            current_speed_embed = torch.where(mask, self.null_speed_embed, current_speed_embed)

        aggregated_visual_tokens = None
        if self.config.image_features:
            current_cam_images = []
            for cam_id, cam_name in enumerate(self.config.image_features.keys()):
                current_cam_images.append(batch["observation.images"][cam_id])
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

            film_params = self.speed_film_gen(current_speed_embed)  # (B, 2*D)
            gamma, beta = torch.split(film_params, self.config.dim_model, dim=1)
            # 广播到 (B, Num_Img_Tokens, D)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            aggregated_visual_tokens = aggregated_visual_tokens * (1 + gamma) + beta

        single_logical_timestep_tokens_list = []

        latent_token = self.encoder_latent_input_proj(latent_sample).unsqueeze(1)
        single_logical_timestep_tokens_list.append(latent_token)

        if self.config.robot_state_feature is not None:
            robot_state_token = self.encoder_robot_state_input_proj(batch["observation.state"]).unsqueeze(1)
            single_logical_timestep_tokens_list.append(robot_state_token)

        if aggregated_visual_tokens is not None:
            single_logical_timestep_tokens_list.append(aggregated_visual_tokens)

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
        global_feature = encoder_out.mean(dim=0)
        pred_speed_logits = self.speed_prediction_head(global_feature)

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

        if return_features:
            return actions, (mu, log_sigma_x2), encoder_out

        return actions, (mu, log_sigma_x2), pred_speed_logits