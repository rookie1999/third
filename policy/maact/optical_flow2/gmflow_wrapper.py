import torch
import torch.nn as nn
import logging

from policy.maact.optical_flow2.gmflow.gmflow import GMFlow

# 引入 GMFlow 类
# 请确保 policy/maact/optical_flow/gmflow/ 目录存在且包含源文件

logger = logging.getLogger(__name__)


class GMFlowWrapper(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()

        logger.info(f"Initializing GMFlowWrapper...")

        # 1. 实例化 GMFlow 模型 (使用标准配置)
        self.model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )

        # 2. 加载预训练权重
        if checkpoint_path:
            logger.info(f"Loading GMFlow checkpoint from {checkpoint_path}")
            # 使用 map_location='cpu' 加载，之后由主模型统一 .to(device)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

            # 加载权重，允许非严格匹配（因为有些辅助用的key可能不需要）
            missing_keys, unexpected_keys = self.model.load_state_dict(weights, strict=False)
            if missing_keys:
                logger.warning(f"GMFlow missing keys: {missing_keys}")
        else:
            logger.warning("No checkpoint path provided for GMFlow! Using random weights.")

        # 3. 冻结参数（光流网络通常只做推理，不参与训练更新）
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image1, image2):
        """
        Args:
            image1: [B, 3, H, W] Tensor, 假设是 [0, 1] 或 [0, 255]
            image2: [B, 3, H, W] Tensor
        Returns:
            flow: [B, 2, H, W]
        """
        with torch.no_grad():
            # 2. 确保模型处于 eval 模式 (例如关闭 Dropout)
            self.model.eval()

            # --- 数据预处理 ---
            if image1.max() <= 1.1:
                img1_input = image1 * 255.0
                img2_input = image2 * 255.0
            else:
                img1_input = image1
                img2_input = image2

            # GMFlow 推理参数
            attn_splits_list = [2]
            corr_radius_list = [-1]
            prop_radius_list = [-1]

            results_dict = self.model(
                img1_input,
                img2_input,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                pred_bidir_flow=False
            )

            flow = results_dict['flow_preds'][-1]  # [B, 2, H, W]

            # 3. 显式 detach，彻底切断梯度流
            return flow.detach()