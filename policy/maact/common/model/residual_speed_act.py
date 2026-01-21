import torch
import torch.nn as nn
from .speed_act_modulate_full_model import get_dim_size


class ResidualSpeedACT(nn.Module):
    def __init__(self, base_act_model, config):
        """
        Args:
            base_act_model: 预训练好并加载了权重的 SpeedACT 实例
            config: 配置对象
        """
        super().__init__()
        self.config = config

        self.base_policy = base_act_model
        for param in self.base_policy.parameters():
            param.requires_grad = False
        self.base_policy.eval()

        action_dim = get_dim_size(config.action_feature, 0)
        self.total_action_dim = config.chunk_size * action_dim

        # 输入：Base Policy 的 Encoder 特征 (dim_model)
        # 输出：动作的残差量 (Chunk * Dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(config.dim_model, 256),
            nn.LayerNorm(256),  # 加个 Norm 会更稳定
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.total_action_dim)
        )

        # Zero Initialization Trick
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # 残差缩放因子 (可作为超参调整)
        # 限制 RL 能修改的最大幅度，例如只允许微调 ±10% 的动作幅度
        self.residual_scale = 0.1

    def forward(self, batch):
        """
        前向传播：
        Action = Base_Action + Scale * Tanh(Residual_Action)
        """
        # 1. 获取 Base Policy 的动作和特征
        with torch.no_grad():
            base_actions, _, _, _, encoder_out = self.base_policy(batch, return_features=True)

        # 2. 提取特征用于残差决策
        # encoder_out: (B, Seq_Len, Dim)
        # 策略A：取第一个 Token (Latent Token)，因为在 Transformer Encoder 中它已经融合了全局信息
        # 策略B：取 Mean Pooling (encoder_out.mean(dim=1))
        residual_input = encoder_out[:, 0, :]

        # 3. 计算残差 (Delta Action)
        # (B, Chunk * Dim)
        delta_actions_flat = self.residual_mlp(residual_input)

        # Reshape 成 (B, Chunk, Dim)
        delta_actions = delta_actions_flat.view(base_actions.shape)

        # 4. 叠加动作
        # 使用 Tanh 限制残差范围在 [-1, 1]，然后乘上 scale
        residual_term = torch.tanh(delta_actions) * self.residual_scale

        # 最终动作
        final_actions = base_actions + residual_term

        return final_actions

    def train(self, mode=True):
        # 重写 train 方法，确保 base_policy 永远保持 eval 模式
        super().train(mode)
        self.base_policy.eval()
        return self