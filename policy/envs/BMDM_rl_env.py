import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np

class BMDMREnv(gym.Env):
    NAME = 'BMDM_ppo'
    """
    针对 BMDM 内部多批次并行的强化学习环境。
    一个 Episode = 一个 Block 从 t=T 到 t=0 的完整去噪序列。
    """
    def __init__(self, config, dataset, physics_reward_system, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.reward_system = physics_reward_system
        self.device = device

        # 参数配置
        self.batch_size = config.get('num_parallel', 64)  # 内部处理多批次
        self.block_size = config['block_size']  # 5
        self.frame_dim = 267
        self.action_dim = self.block_size * self.frame_dim  # 1335
        self.diffusion_steps = config.get('diffusion_steps', 1000)

        # 定义空间 (采用 Batch 维度)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch_size, self.action_dim),
                                       dtype=np.float32)

        # 观测空间包含：当前噪声块, 时间步, 以及作为条件的"前置动作"
        self.observation_space = spaces.Dict({
            'x_t': spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch_size, self.action_dim), dtype=np.float32),
            't': spaces.Box(low=0, high=self.diffusion_steps, shape=(self.batch_size,), dtype=np.int32),
            'cond': spaces.Box(low=-np.inf, high=np.inf, shape=(self.batch_size, self.action_dim), dtype=np.float32)
        })

        # 状态变量
        self.current_t = None
        self.current_x_t = None
        self.current_cond = None

    def reset(self, seed=None, options=None):
        """
        重置环境，开始一批次新的去噪回合 (t=T)
        """
        super().reset(seed=seed)

        # 1. 初始化时间步：所有 batch 同时从 T-1 开始
        self.current_t = torch.full((self.batch_size,), self.diffusion_steps - 1, device=self.device, dtype=torch.long)

        # 2. 随机生成初始高斯噪声 x_T
        self.current_x_t = torch.randn((self.batch_size, self.action_dim), device=self.device)

        # 3. 核心：随机抽取真实的动作作为 cond (前置动作)
        # 假设 dataset 能够返回随机采样的真实动作块
        # [需要补充]: 请确保 dataset 有此批量采样方法
        real_samples = self.dataset.sample_real_blocks(batch_size=self.batch_size)
        self.current_cond = real_samples.to(self.device)  # [Batch, 1335]

        return self._get_obs(), {}

    def step(self, action):
        """
        逐步训练逻辑：
        action: Agent 预测的 x_{t-1} [Batch, 1335]
        """
        # 1. 状态转移
        self.current_x_t = action  # 更新当前的噪声状态
        self.current_t -= 1  # 时间步向前推

        # 2. 判断是否完成 (当时间步减到 0 以下)
        done = bool(self.current_t[0] < 0)  # 批次同步，取第一个即可
        reward = torch.zeros(self.batch_size, device=self.device)
        info = {}

        # 3. 计算物理奖励 (仅在去噪到最后一步 t=0 时计算)
        if done:
            # 执行反归一化 (直接使用您指定的 denorm_data)
            # pose 为生成的当前块，prev_pose 为作为条件的真实前一块
            denorm_pose = self.dataset.denorm_data(self.current_x_t)
            denorm_prev = self.dataset.denorm_data(self.current_cond)

            # 调用物理约束系统计算批量奖励
            # 您的 physics_constraints.py 已支持 [batch_size, 1335] 的计算
            reward_results = self.reward_system.compute_total_reward(
                pose=denorm_pose,
                prev_pose=denorm_prev
            )

            # 获取总奖励张量 [Batch]
            reward = reward_results['total_physics_reward']
            info['physics_details'] = reward_results  # 记录滑步、穿地等明细

        # 返回符合 Gym 规范的 5 元组
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """返回当前的 Tensor 字典观测值"""
        return {
            'x_t': self.current_x_t,
            't': self.current_t,
            'cond': self.current_cond
        }

    def render(self):
        """
        可选：在 t=0 时调用 PBLMocapViewer 展示生成动作
        """
        if self.current_t is not None and self.current_t[0] == 0:
            # 反归一化并渲染
            display_pose = self.dataset.denorm_data(self.current_x_t[0:1])
            # ... 此处可插入 base_env.py 中的 viewer.render 逻辑 ...
            pass