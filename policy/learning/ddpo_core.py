"""
DDPO (Denoising Diffusion Policy Optimization) 核心模块
将BMDM作为可训练的策略，通过强化学习微调其去噪偏好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DenoisingTrajectory:
    """
    去噪轨迹记录
    记录从T到0的每一步去噪过程
    """
    x_t: torch.Tensor
    t: int
    pred_epsilon: torch.Tensor
    pred_x0: torch.Tensor
    x_prev: torch.Tensor
    log_prob: Optional[torch.Tensor] = None


class DDPOBuffer:
    """
    DDPO经验缓冲区
    存储去噪轨迹和对应的奖励
    """

    def __init__(self, buffer_size: int, num_diffusion_steps: int, batch_size: int, action_dim: int):
        self.buffer_size = buffer_size
        self.num_diffusion_steps = num_diffusion_steps
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.trajectories = []
        self.rewards = []
        self.conditions = []
        self.log_probs = []

        self.ptr = 0
        self.size = 0

    def add_trajectory(
        self,
        trajectory: List[DenoisingTrajectory],
        reward: torch.Tensor,
        condition: torch.Tensor,
        log_probs: List[torch.Tensor]
    ):
        if len(self.trajectories) >= self.buffer_size:
            self.trajectories.pop(0)
            self.rewards.pop(0)
            self.conditions.pop(0)
            self.log_probs.pop(0)

        self.trajectories.append(trajectory)
        self.rewards.append(reward)
        self.conditions.append(condition)
        self.log_probs.append(log_probs)
        self.size = min(self.size + 1, self.buffer_size)

    def clear(self):
        self.trajectories = []
        self.rewards = []
        self.conditions = []
        self.log_probs = []
        self.size = 0

    def get_batch(self, batch_size: int) -> Dict:
        indices = torch.randint(0, self.size, (min(batch_size, self.size),))

        batch_trajectories = [self.trajectories[i] for i in indices]
        batch_rewards = torch.stack([self.rewards[i] for i in indices])
        batch_conditions = torch.stack([self.conditions[i] for i in indices])
        batch_log_probs = [self.log_probs[i] for i in indices]

        return {
            'trajectories': batch_trajectories,
            'rewards': batch_rewards,
            'conditions': batch_conditions,
            'log_probs': batch_log_probs
        }


class DDPOSampler:
    """
    DDPO采样器
    执行去噪过程并记录轨迹
    """

    def __init__(self, diffusion_model, config):
        self.diffusion = diffusion_model
        self.T = diffusion_model.T
        self.estimate_mode = diffusion_model.estimate_mode
        self.device = next(diffusion_model.parameters()).device

        self.ddpo_config = config.get('ddpo', {})
        self.noise_scale = self.ddpo_config.get('noise_scale', 1.0)
        self.clip_sample = self.ddpo_config.get('clip_sample', True)
        self.clip_range = self.ddpo_config.get('clip_range', (-1.0, 1.0))

    def _robust_extract(self, tensor_param, ts):
        # 无论 tensor_param 是 [T], [T,1] 还是 [1,T]，全压平为 1D，再用索引取值
        val = tensor_param.view(-1)[ts.long()]
        return val.view(-1, 1)  # 变回 [batch_size, 1] 方便与特征广播相乘

    def sample_with_trajectory(
        self,
        condition: torch.Tensor,
        extra_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, List[DenoisingTrajectory]]:
        """
        执行去噪采样并记录完整轨迹
        condition: [batch_size, condition_dim]
        return: final_x, trajectory_list
        """
        batch_size = condition.shape[0]
        x = torch.randn(batch_size, self.diffusion.frame_dim, device=self.device)

        trajectory = []

        cond_emb = None
        if self.diffusion.conds_flag and extra_info is not None:
            traj_pose = extra_info.get("traj_pose", None)
            traj_trans = extra_info.get("traj_trans", None)
            if traj_pose is not None and traj_trans is not None:
                cond = torch.cat([traj_pose, traj_trans], dim=-1)
                cond_emb = self.diffusion.cond_proj(cond)

        for t in range(self.T - 1, -1, -1):
            ts = torch.tensor([t], device=self.device).repeat(batch_size)
            ts = ts.to(torch.int64)
            te = self.diffusion.time_mlp(ts)

            if cond_emb is not None:
                latent = torch.cat((te, cond_emb), dim=-1)
            else:
                latent = te

            pred = self.diffusion.model(condition, x, latent)

            pred_x0 = None
            # if self.estimate_mode == 'epsilon':   ###
            #     pred_x0 = self.diffusion.get_x0_from_xt(x, ts, pred)
            # elif self.estimate_mode == 'x0':
            #     pred_x0 = pred
            
            # if self.clip_sample and pred_x0 is not None:
            #     pred_x0 = torch.clamp(pred_x0, self.clip_range[0], self.clip_range[1])

            # log_prob = self._compute_log_prob(x, pred, ts)

            # traj_step = DenoisingTrajectory(
            #     x_t=x.clone(),
            #     t=t,
            #     pred_epsilon=pred if self.estimate_mode == 'epsilon' else None,
            #     pred_x0=pred_x0,
            #     log_prob=log_prob
            # )
            # trajectory.append(traj_step)

            # if self.estimate_mode == 'epsilon':
            #     x = self.diffusion.remove_noise(x, pred, ts)
            # elif self.estimate_mode == 'x0':
            #     x = pred

            # if self.clip_sample:
            #     x = torch.clamp(x, self.clip_range[0], self.clip_range[1])

            # if t > 0:
            #     x = self.diffusion.add_noise(x, ts)
            if self.estimate_mode == 'epsilon':
                # 注意：确保你的 remove_noise 函数只返回分布的均值(mu_t)！
                # 如果它内部自己加了噪声，你需要把它剥离，因为RL必须自己掌控采样噪声！
                mu_t = self.diffusion.remove_noise(x, pred, ts)
            elif self.estimate_mode == 'x0':
                pred_x0 = pred
                # 必须计算后验均值，而不是直接拿 pred 当 x！
                # 这里调用你的 diffusion 类里的后验均值计算函数 (可能叫 q_posterior 或直接手写)
                mu_t = self._q_posterior_mean(x_0_pred=pred_x0, x_t=x, ts=ts)
            # ---------------------------------------------------------
            # 第二步：获取方差并执行 RL 的探索采样 (Action)
            # ---------------------------------------------------------
            sigma = self.diffusion.extract(self.diffusion.sigma, ts, x.shape)
            # sigma = sigma.to(torch.int64)
            # print("sigma", sigma.shape)
            
            if t > 0:
                # 强化学习的探索性 (Entropy) 就是从这个 randn 来的！
                noise = torch.randn_like(x) 
                x_prev = mu_t + sigma * noise
            else:
                x_prev = mu_t  # 最后一步不需要噪声

            if self.clip_sample:
                x_prev = torch.clamp(x_prev, self.clip_range[0], self.clip_range[1])

            # ---------------------------------------------------------
            # 第三步：利用真正的 RL Action 计算 Log Probability
            # ---------------------------------------------------------
            log_prob = self._compute_log_prob(x_prev, mu_t, sigma)

            traj_step = DenoisingTrajectory(
                x_t=x.clone(),
                t=t,
                pred_epsilon=pred if self.estimate_mode == 'epsilon' else None,
                pred_x0=pred_x0,
                x_prev=x_prev,
                log_prob=log_prob
            )
            trajectory.append(traj_step)

            # 更新 x 为真正的 x_{t-1}，进入下一次迭代
            x = x_prev

        return x, trajectory

    def _compute_log_prob(self, x_prev: torch.Tensor, mu_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算当前去噪步骤的对数概率
        使用高斯分布假设
        """
        # sigma = self.diffusion.extract(self.diffusion.sigma, ts, x_t.shape)
        # # log_prob = -0.5 * ((x_t - pred) ** 2) / (sigma ** 2 + 1e-8)
        # # log_prob = log_prob.sum(dim=-1, keepdim=True)
        # # return log_prob
        #
        # variance = sigma ** 2 + 1e-8
        #
        # # 公式: -0.5 * ((x - mu)^2 / var + log(2 * pi * var))
        # log_prob = -0.5 * ((x_prev - mu_t) ** 2) / variance - 0.5 * torch.log(2 * torch.pi * variance)
        #
        # # 沿着特征维度求和，得到针对单个样本的标量概率
        # return log_prob.sum(dim=-1, keepdim=True)
        variance = sigma ** 2 + 1e-8
        variance = variance.view(x_prev.shape[0], -1)

        # print("x_prev, mu_t, variance", x_prev.shape, mu_t.shape, variance.shape)
        # 公式: -0.5 * ((x - mu)^2 / var + log(2 * pi * var))
        log_prob = -0.5 * ((x_prev - mu_t) ** 2) / variance - 0.5 * torch.log(2 * torch.pi * variance)

        ###
        # 简化后的伪 log_prob，只计算均方误差的负值
        log_prob = -0.5 * ((x_prev - mu_t) ** 2)

        # 沿着特征维度求和，得到针对单个样本的标量概率
        return log_prob.sum(dim=-1, keepdim=True)

    def _q_posterior_mean(self, x_0_pred: torch.Tensor, x_t: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        计算 DDPM 在给定预测 x_0 和当前 x_t 下的后验均值 mu_t
        """
        extract = self.diffusion.extract
        
        # 提取当前时间步的扩散参数
        # 注意：这里假设您的 diffusion 对象中有 alphas, betas, alphas_cumprod 等标准属性
        betas_t = extract(self.diffusion.betas, ts, x_t.shape)
        alphas_t = extract(self.diffusion.alphas, ts, x_t.shape)
        alphas_cumprod_t = extract(self.diffusion.alphas_cumprod, ts, x_t.shape)
        
        # 提取上一时间步的累积乘积 (alphas_cumprod_prev)
        if hasattr(self.diffusion, 'alphas_cumprod_prev'):
            alphas_cumprod_prev_t = extract(self.diffusion.alphas_cumprod_prev, ts, x_t.shape)
        else:
            # 如果没有现成的 prev，就手动将 alphas_cumprod 向右平移一位，首位补 1.0
            alphas_cumprod_prev = torch.cat([
                torch.tensor([1.0], device=x_t.device), 
                self.diffusion.alphas_cumprod[:-1]
            ])
            alphas_cumprod_prev_t = extract(alphas_cumprod_prev, ts, x_t.shape)

        # 核心公式计算
        weight_x0 = (betas_t * torch.sqrt(alphas_cumprod_prev_t)) / (1.0 - alphas_cumprod_t)
        weight_xt = ((1.0 - alphas_cumprod_prev_t) * torch.sqrt(alphas_t)) / (1.0 - alphas_cumprod_t)
        
        # 组合得到最终均值
        mu_t = weight_x0 * x_0_pred + weight_xt * x_t
        return mu_t

class DDPOPolicy(nn.Module):
    """
    DDPO策略包装器
    将BMDM模型包装为可训练的策略
    """

    def __init__(self, bmdm_model, config):
        super().__init__()
        self.bmdm = bmdm_model
        self.diffusion = bmdm_model.diffusion
        self.T = bmdm_model.T
        self.frame_dim = bmdm_model.frame_dim
        self.block_size = bmdm_model.block_size

        self.sampler = DDPOSampler(self.diffusion, config)
        self.ddpo_config = config.get('ddpo', {})

        self.train_diffusion_steps = self.ddpo_config.get('train_diffusion_steps', None)
        if self.train_diffusion_steps is None:
            self.train_diffusion_steps = list(range(self.T))

    def forward(self, condition: torch.Tensor, extra_info: Optional[Dict] = None):
        """
        前向传播：执行去噪采样
        """
        return self.sampler.sample_with_trajectory(condition, extra_info)

    def get_parameters(self):
        """获取可训练参数"""
        return self.bmdm.parameters()

    def compute_new_log_prob(
        self,
        trajectory: List[DenoisingTrajectory],
        condition: torch.Tensor,
        extra_info: Optional[Dict] = None
    ) -> List[torch.Tensor]:
        """
        重新计算轨迹的对数概率（用于PPO更新）
        """
        new_log_probs = []

        cond_emb = None
        if self.diffusion.conds_flag and extra_info is not None:
            traj_pose = extra_info.get("traj_pose", None)
            traj_trans = extra_info.get("traj_trans", None)
            if traj_pose is not None and traj_trans is not None:
                cond = torch.cat([traj_pose, traj_trans], dim=-1)
                cond_emb = self.diffusion.cond_proj(cond)

        for step in trajectory:
            if step.t not in self.train_diffusion_steps:
                new_log_probs.append(step.log_prob)
                continue

            ts = torch.tensor([step.t], device=step.x_t.device).repeat(step.x_t.shape[0])
            ts = ts.to(torch.int64)
            te = self.diffusion.time_mlp(ts)

            if cond_emb is not None:
                latent = torch.cat((te, cond_emb), dim=-1)
            else:
                latent = te

            pred = self.diffusion.model(condition, step.x_t, latent)

            # 2. 修复：必须严格还原均值 mu_t 的计算！
            if self.sampler.estimate_mode == 'epsilon':
                mu_t = self.diffusion.remove_noise(step.x_t, pred, ts)
            elif self.sampler.estimate_mode == 'x0':
                mu_t = self.sampler._q_posterior_mean(x_0_pred=pred, x_t=step.x_t, ts=ts)
            sigma = self.diffusion.extract(self.diffusion.sigma, ts, step.x_t.shape)

            new_log_prob = self.sampler._compute_log_prob(step.x_prev, mu_t, sigma)
            new_log_probs.append(new_log_prob)

        return new_log_probs


class DDPORewardCalculator:
    """
    DDPO奖励计算器
    计算物理约束奖励
    """

    def __init__(self, config, dataset, device):
        self.config = config
        self.dataset = dataset
        self.device = device

        from policy.learning.physics_constraints import PhysicsConstraintReward
        self.physics_reward = PhysicsConstraintReward(config, dataset, device)

        self.reward_config = config.get('ddpo', {}).get('reward', {})
        self.physics_weight = self.reward_config.get('physics_weight', 1.0)
        self.quality_weight = self.reward_config.get('quality_weight', 0.1)
        self.diversity_weight = self.reward_config.get('diversity_weight', 0.1)

    def compute_reward(
        self,
        generated_motion: torch.Tensor,
        condition: torch.Tensor,
        prev_motion: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总奖励
        generated_motion: [batch_size, frame_dim]
        return: total_reward, reward_dict
        """
        rewards = {}

        physics_reward, physics_components = self.physics_reward(generated_motion, prev_motion)
        rewards['physics'] = physics_reward
        rewards['physics_components'] = physics_components

        quality_reward = self._compute_quality_reward(generated_motion, condition)
        rewards['quality'] = quality_reward

        total_reward = (
            self.physics_weight * physics_reward +
            self.quality_weight * quality_reward
        )
        total_reward = total_reward.squeeze(-1)
        rewards['total'] = total_reward

        return total_reward, rewards

    def _compute_quality_reward(self, generated: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        计算生成质量奖励
        基于生成结果与条件的连贯性
        """
        block_size = self.config.get('block_size', 5)
        single_dim = int(generated.shape[-1] / block_size)

        condition_reshaped = condition.reshape(-1, block_size, single_dim)
        generated_reshaped = generated.reshape(-1, block_size, single_dim)

        last_frame_cond = condition_reshaped[:, -1, :]
        first_frame_gen = generated_reshaped[:, 0, :]

        continuity_reward = -torch.norm(first_frame_gen - last_frame_cond, dim=-1, keepdim=True)

        return continuity_reward
