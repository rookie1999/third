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

        # 1. 冻结 Base Policy
        self.base_policy = base_act_model
        for param in self.base_policy.parameters():
            param.requires_grad = False
        self.base_policy.eval()

        # 2. 计算 MLP 输入维度
        action_dim = get_dim_size(config.action_feature, 0)
        self.total_action_dim = config.chunk_size * action_dim

        # 确定要使用的 Token 数量
        # 我们至少使用 Latent(0) 和 Speed Token
        # 如果有 Robot State，也加上
        self.feature_indices = [0]  # Always include Latent Token (Index 0)
        current_idx = 1

        if self.config.robot_state_feature is not None:
            self.feature_indices.append(current_idx)  # Include Robot State Token
            current_idx += 1

        self.feature_indices.append(current_idx)  # Include Speed Token
        # 注意: Speed Token 紧跟在 Robot State 之后 (参考 speed_act_with_speed.py 的构建顺序)

        # MLP 输入维度 = 选中的Token数 * 模型维数 + 基础动作维数
        num_context_tokens = len(self.feature_indices)
        input_dim = (num_context_tokens * config.dim_model) + self.total_action_dim

        # 3. 定义残差网络 (MLP)
        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.total_action_dim)
        )

        # 4. Zero Initialization Trick (关键)
        # 确保初始状态下残差为 0，不破坏 Base Policy 的表现
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # 残差缩放因子
        self.residual_scale = 0.1

    def forward(self, batch):
        """
        前向传播：
        Action = Base_Action + Scale * Tanh(Residual_Action)
        """
        # 1. 获取 Base Policy 的动作和特征 (No Grad)
        with torch.no_grad():
            base_actions, _, encoder_out = self.base_policy(batch, return_features=True)

        # 2. 提取特征
        # 注意：encoder_out 的形状通常是 (Seq_Len, Batch, Dim)
        # 我们需要转置为 (Batch, Seq_Len, Dim) 以便索引，或者直接按维度 0 索引

        # 收集关键 Token: Latent, RobotState, Speed
        # encoder_out[idx] 取出的是 (Batch, Dim)
        selected_features = []
        for idx in self.feature_indices:
            if idx < encoder_out.shape[0]:
                selected_features.append(encoder_out[idx])
            else:
                raise IndexError(
                    f"Feature index {idx} out of bounds for encoder output with len {encoder_out.shape[0]}")

        # 拼接特征: (B, Num_Tokens * Dim)
        context_features = torch.cat(selected_features, dim=-1)

        # 3. 准备 MLP 输入
        # 将 Base Actions 展平: (B, Chunk, Dim) -> (B, Chunk * Dim)
        base_actions_flat = base_actions.view(base_actions.shape[0], -1)

        # 拼接上下文特征和原始动作
        # 这样 MLP 既知道“环境状况”(features) 也知道“原计划”(base_actions)
        residual_input = torch.cat([context_features, base_actions_flat], dim=-1)

        # 4. 计算残差
        delta_actions_flat = self.residual_mlp(residual_input)

        # Reshape 回动作形状 (B, Chunk, Dim)
        delta_actions = delta_actions_flat.view(base_actions.shape)

        # 5. 叠加动作
        # 使用 Tanh 限制残差范围在 [-1, 1]，然后乘上 scale
        residual_term = torch.tanh(delta_actions) * self.residual_scale

        final_actions = base_actions + residual_term

        return final_actions

    def train(self, mode=True):
        """
        重写 train 方法，确保 base_policy 永远保持 eval 模式
        """
        super().train(mode)
        self.base_policy.eval()
        return self