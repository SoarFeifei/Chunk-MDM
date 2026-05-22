import torch.nn as nn
import torch
from policy.common.controller import init, DiagGaussian_adaptive, DiagGaussian_fixed


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=0.01)  ### zero_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)  ### torch.nn.init.zero_(m.bias)


class PPOModel(nn.Module):
    NAME = 'PPO'

    def __init__(self, config, env, device):
        super().__init__()

        self.actor = ActorNet(env).to(device)
        self.critic = CriticNet(env).to(device)
        init_weights(self.actor)
        self.distr_type = config['distr_type']
        self.std_value = config['distr_std']
        self.block_size = config['block_size']

        if self.distr_type == 'fixed':
            self.dist = DiagGaussian_fixed(self.std_value)
        elif self.distr_type == 'adaptive':
            self.dist = DiagGaussian_adaptive(self.actor.action_dim)

        self.state_size = 1

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)  ### inputs torch.Size([512, 269])   267 * n + 2
        # print("action",action.shape)    ### action torch.Size([512, 3738]) 3738 = 267*14 * block_size
        # action = action * torch.min(self.w, 1.0)

        dist = self.dist(action)
        if deterministic:
            action = dist.mode()

        else:
            action = dist.sample()
            # action.clamp_(-3.0, 3.0)    ###(-1.0, 1.0) $$
        action_log_probs = dist.log_probs(action)
        # print("action_log_probs", action_log_probs.shape) # torch.Size([512, 1])

        ###
        # inputs = inputs[:, -269:]
        value = self.critic(inputs)

        return value, action, action_log_probs  ###  / action.shape[1]

    def get_value(self, inputs):
        ###
        # inputs = inputs[:, -269:]
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        # value = self.critic(inputs[:, -269:])   ###
        value = self.critic(inputs)
        mode = self.actor(inputs)
        if torch.isnan(value).any():
            print("inputs", inputs)
            print("value=== 检测到 NaN ===")
            print("value", value)
        if torch.isnan(mode).any():
            print("mode=== 检测到 NaN ===")
            print("mode", mode)
        # mode = mode * torch.min(self.w, 1.0)

        dist = self.dist(mode)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy  ###  / action.shape[1]


class CriticNet(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dim = env.observation_space.shape[0]
        # self.observation_dim = int(env.frame_dim / env.block_size) + 2 ###
        h_size = 256
        self.single_dim = int(env.frame_dim / env.block_size)
        self.target_dim = 2
        self.block_size = env.block_size
        # self.critic = nn.Sequential(
        #         nn.Linear(self.observation_dim, h_size),
        #         # nn.LayerNorm(h_size),  # 添加层归一化     ###
        #         nn.ReLU(),
        #         nn.Linear(h_size, h_size),
        #         # nn.LayerNorm(h_size),  # 添加层归一化
        #         nn.ReLU(),
        #         nn.Linear(h_size, h_size),
        #         # nn.LayerNorm(h_size),  # 添加层归一化
        #         nn.ReLU(),
        #         # nn.Linear(h_size, h_size),  ###
        #         # # nn.LayerNorm(h_size),  # 添加层归一化
        #         # nn.ReLU(),
        #         nn.Linear(h_size, 1)
        #     )

        # 共享权重可以显著减少参数量，提高泛化性
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.single_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 2. 目标编码器：处理 2D 目标信息
        # self.target_encoder = nn.Sequential(
        #     nn.Linear(self.target_dim, 64),
        #     nn.ReLU()
        # )

        # 3. 融合层：整合历史块特征 (128 * 5) + 目标特征 (64)
        fusion_input_dim = (128 * self.block_size)  # + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出单一 Value
        )

    def forward(self, x):
        ###
        # # x = x[:, -269:]
        # return self.critic(x)
        batch_size = x.shape[0]

        # --- A. 数据拆分 ---
        # 提取历史动作块 (Batch, 5 * 267)
        history_motion = x[:, :self.single_dim * self.block_size]
        # 提取目标 (Batch, 2)
        # target = x[:, -self.target_dim:]

        # --- B. 特征编码 ---
        # 将历史动作重构为 (Batch * block_size, frame_dim) 进行批处理编码
        history_frames = history_motion.reshape(batch_size * self.block_size, self.single_dim)
        frame_feats = self.frame_encoder(history_frames)  # (Batch * 5, 128)

        # 还原并打平特征 (Batch, 5 * 128)
        history_feat_flat = frame_feats.reshape(batch_size, -1)

        # 编码目标特征 (Batch, 64)
        # target_feat = self.target_encoder(target)

        # --- C. 信息融合 ---
        # combined = torch.cat([history_feat_flat, target_feat], dim=-1)
        latent_context = self.fusion_layer(history_feat_flat)  # (Batch, 512)    ### combined

        # --- D. 生成引导块 (Chunk) ---
        value = self.value_head(latent_context)  # (Batch, 1)
        return value


class ActorNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.device = env.device
        self.observation_dim = env.observation_space.shape[0]  ### 512, 269    803 = 267 * 3 +2
        print("self.observation_dim", self.observation_dim)
        self.action_dim = env.action_space.shape[0]
        print("action_dim", env.action_dim)
        print("self.action_dim", self.action_dim)  ### self.action_dim 3738 1335
        # h_size = 256
        # self.actor = nn.Sequential(
        #     nn.Linear(self.observation_dim, h_size),    # self.observation_dim
        #     # nn.LayerNorm(h_size),  # 添加层归一化     ###
        #     nn.ReLU(),
        #     nn.Linear(h_size, h_size),
        #     # nn.LayerNorm(h_size),  # 添加层归一化
        #     nn.ReLU(),
        #     nn.Linear(h_size, h_size),
        #     # nn.LayerNorm(h_size),  # 添加层归一化
        #     nn.ReLU(),
        #     # nn.Linear(h_size, h_size),  ###
        #     # # nn.LayerNorm(h_size),  # 添加层归一化
        #     # nn.ReLU(),
        #     nn.Linear(h_size, self.action_dim),
        #     nn.Tanh()
        # )
        ###
        self.block_size = env.block_size
        self.frame_dim = env.frame_dim
        self.single_dim = int(env.frame_dim / env.block_size)
        print("frame_dim", self.frame_dim, self.block_size, self.single_dim)
        self.target_dim = env.observation_space.shape[0] - self.frame_dim
        print("self.target_dim", self.target_dim)

        # 1. 帧编码器：独立处理每一帧动作，提取局部运动特征
        # 共享权重可以显著减少参数量，提高泛化性
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.single_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 2. 目标编码器：处理 2D 目标信息
        # self.target_encoder = nn.Sequential(
        #     nn.Linear(self.target_dim, 64),
        #     nn.ReLU()
        # )

        # 3. 融合层：整合历史块特征 (128 * 5) + 目标特征 (64)
        fusion_input_dim = (128 * self.block_size)  # + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        # 4. Chunk 解码器：输出未来 5 帧的控制信号
        # 输出维度为 block_size * frame_dim (5 * 267 = 1335)
        self.chunk_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_dim),
            nn.Tanh()  # $$
        )

    def forward(self, x):
        # print("x", x.shape)     ### x torch.Size([512, 269])
        ###
        # return self.actor(x)
        # return action
        # state 形状: (Batch, frame_dim * block_size + target_dim)
        batch_size = x.shape[0]

        # --- A. 数据拆分 ---
        # 提取历史动作块 (Batch, 5 * 267)
        history_motion = x[:, :self.single_dim * self.block_size]
        # 提取目标 (Batch, 2)
        # target = x[:, -self.target_dim:]

        # --- B. 特征编码 ---
        # 将历史动作重构为 (Batch * block_size, frame_dim) 进行批处理编码
        history_frames = history_motion.reshape(batch_size * self.block_size, self.single_dim)
        frame_feats = self.frame_encoder(history_frames)  # (Batch * 5, 128)
        # 还原并打平特征 (Batch, 5 * 128)
        history_feat_flat = frame_feats.reshape(batch_size, -1)

        # 编码目标特征 (Batch, 64)
        # target_feat = self.target_encoder(target)

        # --- C. 信息融合 ---
        # combined = torch.cat([history_feat_flat, target_feat], dim=-1)
        latent_context = self.fusion_layer(history_feat_flat)  # (Batch, 512)   ###combined

        # --- D. 生成引导块 (Chunk) ---
        chunk_output = self.chunk_decoder(latent_context)  # (Batch, 1335)
        return chunk_output
        # 注意：这里保持打平输出以适配 PPO 的动作分布处理
        # 在 env.step 中再进行 reshape(5, 267)

