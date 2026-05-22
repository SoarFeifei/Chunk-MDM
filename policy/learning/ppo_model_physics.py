import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import model.modules.Embedding as Embedding
import model.modules.Activation as Activation

class ActorNet(nn.Module):
    """
    Actor 网络：利用预训练 BMDM 进行动作分布的均值预测
    """

    def __init__(self, bmdm_model, action_dim=1335, log_std_init=-0.5):
        super().__init__()
        # 1. 核心：封装预训练的 BMDM
        self.bmdm = bmdm_model

        # 2. RL 探索方差：action_dim 为 5 * 267 = 1335
        # 使用可学习参数，为每个动作维度提供探索空间
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, x_t, t, cond):
        """
        x_t: 当前噪声块 [Batch, 1335]
        t: 当前扩散步 [Batch]
        cond: 条件输入 (包含上一块动作的向量表示 [Batch, 1335])
        """
        # 调用您指定的接口获取去噪均值 mu
        # bmdm.sample_rl_ddpm 应返回对应 t 步去噪后的动作均值预测
        mu = self.bmdm.sample_rl_ddpm(x_t, t, cond)

        # 构建高斯策略分布
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist, mu


class CriticNet(nn.Module):
    """
    Critic 网络：完全独立的轻量级 MLP，评估 (x_t, t, cond) 的价值
    """

    def __init__(self, action_dim=1335, cond_dim=1335, time_emb_dim=128, hidden_dim=512):
        super().__init__()

        # 1. 时间步位置编码层
        self.time_mlp = torch.nn.Sequential(
            Embedding.PositionalEmbedding(self.time_emb_dim, 1.0),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
            Activation.SiLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        # 2. MLP 骨干网络
        # 输入维度 = 当前块(1335) + 时间步嵌入(128) + 条件向量(1335)
        input_dim = action_dim + time_emb_dim + cond_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出 V 标量
        )

    def forward(self, x_t, t, cond):
        # 获取时间步的位置编码嵌入
        t_emb = self.time_mlp(t)  # [Batch, 128]

        # 拼接所有输入信息进行评估
        # 确保 cond 是原始向量表示，与 x_t 在维度上对齐
        h = torch.cat([x_t, t_emb, cond], dim=-1)

        return self.mlp(h)


class PPOModel(nn.Module):
    """
    PPO 策略包装类，供 PPOAgent 调用
    """

    def __init__(self, config, bmdm_model, env, time_emb_dim=64):
        super().__init__()
        action_dim = env.observation_space.shape[0]
        cond_dim = env.observation_space.shape[0]
        self.actor = ActorNet(bmdm_model, action_dim)
        self.critic = CriticNet(action_dim, cond_dim, time_emb_dim)

        self.distr_type = config['distr_type']
        self.std_value = config['distr_std']
        self.block_size = config['block_size']

        if self.distr_type == 'fixed':
            self.dist = DiagGaussian_fixed(self.std_value)
        elif self.distr_type == 'adaptive':
            self.dist = DiagGaussian_adaptive(self.actor.action_dim)

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)

        dist = self.dist(action)
        if deterministic:
            action = dist.mode()

        else:
            action = dist.sample()
            action.clamp_(-1.0, 1.0)
        action_log_probs = dist.log_probs(action)

        ###
        inputs = inputs[:, -269:]
        value = self.critic(inputs)

        return value, action, action_log_probs ###  / action.shape[1]

    def get_value(self, x_t, t, cond):
        return self.critic(x_t, t, cond)

    def get_action_and_value(self, x_t, t, cond, action=None):
        dist, mu = self.actor(x_t, t, cond)
        value = self.critic(x_t, t, cond)

        if action is None:
            action = dist.sample()

        # 计算 log 概率，用于 PPO 的重要性采样比率
        log_prob = dist.log_prob(action).sum(dim=-1)
        # 计算熵，用于增加探索性
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value