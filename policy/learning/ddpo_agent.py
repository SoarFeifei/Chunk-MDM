"""
DDPO Agent - Denoising Diffusion Policy Optimization Agent
直接微调BMDM模型，使其生成符合物理约束的动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional

from policy.learning.ddpo_core import DDPOBuffer, DDPOPolicy, DDPOSampler, DDPORewardCalculator, DenoisingTrajectory
from policy.learning.physics_constraints import PhysicsConstraintReward
import util.logging as logging_util
import util.save as save_util


class DDPOAgent:
    """
    DDPO Agent
    将BMDM作为策略进行强化学习微调
    """
    NAME = 'DDPO'

    def __init__(self, config, bmdm_model, env, device):
        self.config = config
        self.device = device
        self.env = env
        self.bmdm = bmdm_model

        self.ddpo_config = config.get('ddpo', {})

        self.T = bmdm_model.T
        self.frame_dim = bmdm_model.frame_dim
        self.block_size = bmdm_model.block_size

        self.policy = DDPOPolicy(bmdm_model, config)
        self.policy.to(device)

        self.reward_calculator = DDPORewardCalculator(config, env.dataset, device)

        self.lr = self.ddpo_config.get('lr', 1e-6)
        self.clip_param = self.ddpo_config.get('clip_param', 0.2)
        self.ppo_epochs = self.ddpo_config.get('ppo_epochs', 4)
        self.mini_batch_size = self.ddpo_config.get('mini_batch_size', 64)
        self.value_loss_coef = self.ddpo_config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.ddpo_config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.ddpo_config.get('max_grad_norm', 1.0)
        self.gamma = self.ddpo_config.get('gamma', 0.99)
        self.gae_lambda = self.ddpo_config.get('gae_lambda', 0.95)

        self.num_samples_per_epoch = self.ddpo_config.get('num_samples_per_epoch', 64)
        self.save_interval = self.ddpo_config.get('save_interval', 100)

        self.optimizer = optim.Adam(
            self.policy.get_parameters(),
            lr=self.lr,
            eps=1e-8
        )

        self.lr_decay_type = config.get('lr_decay_type', 'constant')
        self.final_lr = config.get('final_lr', self.lr * 0.1)

        self.buffer = DDPOBuffer(
            buffer_size=self.ddpo_config.get('buffer_size', 1000),
            num_diffusion_steps=self.T,
            batch_size=self.mini_batch_size,
            action_dim=self.frame_dim
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.frame_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        self.value_optimizer = optim.Adam(self.value_head.parameters(), lr=self.lr)

        # if not env.is_rendered:
        self.logger = logging_util.wandbLogger(
            run_name=env.int_output_dir,
            proj_name="DDPO_{}_{}_{}".format(env.NAME, env.model.NAME, env.dataset.NAME)
        )

        self.prev_motion = None
        self.update_count = 0

    def collect_samples(self, num_samples: int) -> Dict:
        """
        收集样本：执行去噪采样并计算奖励
        """
        rl_list = [[60443], [6983], [78604], [33503], [91375],
                [24275], [65582], [63029], [95138], [10771],
                [44264], [32127], [92873], [75495], [48956], [43221]]
        rl_list = np.array(rl_list)

        # rl_list = [x[0] for x in lsit_1]
        self.env.reset(rl_list=rl_list)

        samples = {
            'trajectories': [],
            'rewards': [],
            'conditions': [],
            'generated_motions': [],
            'log_probs': [],
            'reward_dict': []
        }
        # print("history", self.env.history.shape)
        # print("num_parallel", self.env.num_parallel)
        for _ in range(num_samples):
            condition = self.env.get_cond_frame()

            with torch.no_grad():
                generated_motion, trajectory = self.policy(condition, self.env.cur_extra_info)

            # print("generated_motion, condition", generated_motion.shape, condition.shape)
            reward, reward_dict = self.reward_calculator.compute_reward(
                generated_motion, condition, self.prev_motion
            )

            self.env.step_rl(generated_motion)

            log_probs = [step.log_prob for step in trajectory]

            samples['trajectories'].append(trajectory)
            samples['rewards'].append(reward)
            samples['conditions'].append(condition)
            samples['generated_motions'].append(generated_motion)
            samples['log_probs'].append(log_probs)
            samples['reward_dict'].append(reward_dict['physics_components'])

            self.prev_motion = generated_motion

        return samples

    def compute_advantages(self, rewards: List[torch.Tensor], values: torch.Tensor) -> torch.Tensor:
        """
        计算GAE优势函数
        DDPO 的每一次生成是完全独立的 Episode，不存在时序 MDP 转移。
        优势 = 真实奖励 - 价值网络预测值
        rewards: [num_samples, num_parallel]
        values: [num_samples, num_parallel]
        """
        # advantages = []
        # gae = 0
        #
        # for t in reversed(range(len(rewards))):
        #     if t == len(rewards) - 1:
        #         next_value = 0
        #     else:
        #         next_value = values[t + 1]
        #
        #     delta = rewards[t] + self.gamma * next_value - values[t]
        #     gae = delta + self.gamma * self.gae_lambda * gae
        #     advantages.insert(0, gae)
        #
        # advantages = torch.cat(advantages, dim=0)

        # 直接计算优势 (Reward - Value)
        advantages = rewards - values

        # 优势函数归一化（极其重要，稳定 PPO 训练的基石）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def compute_ppo_loss(
        self,
        old_log_probs: List[torch.Tensor],
        new_log_probs: List[torch.Tensor],
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        计算PPO损失
        """
        total_loss = 0
        num_trajectories = len(old_log_probs)

        # for old_lp, new_lp in zip(old_log_probs, new_log_probs):
        #     if old_lp is None or new_lp is None:
        #         continue
        #     old_lp = torch.tensor(old_lp, device='cuda:0')
        #     new_lp = torch.tensor(new_lp, device='cuda:0')
        #     ratio = torch.exp(new_lp - old_lp.detach())
        #
        #     surr1 = ratio * advantages.detach().unsqueeze(-1)   ### advantages.unsqueeze(-1)
        #     surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach().unsqueeze(-1)
        #
        #     step_loss = -torch.min(surr1, surr2).mean()
        #     total_loss += step_loss
        #     num_steps += 1
        for i in range(num_trajectories):
            # 修复 1：必须使用 torch.stack 保留 BMDM 网络的梯度图！
            # old_lp 不需要梯度，所以加 detach()
            old_lp = torch.stack(old_log_probs[i]).to(self.device).detach()
            new_lp = torch.stack(new_log_probs[i]).to(self.device)

            # print("old_lp", old_lp.shape)

            # 修复 2：DDPO 是轨迹级别的强化学习
            # 必须先将 T 步的对数概率求和，再进行 exp，否则步级别的误差极易导致梯度爆炸或消失 ###
            # sum_old_lp = old_lp.sum(dim=0)
            # sum_new_lp = new_lp.sum(dim=0)

            diff = new_lp - old_lp
            min_before = diff.min().item()
            max_before = diff.max().item()
            diff = torch.clamp(diff, min=-20.0, max=5.0)
            ratio = torch.exp(diff)

            adv = advantages[i].detach().view(-1, 1)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv

            step_loss = -torch.min(surr1, surr2).mean()  ### -
            total_loss += step_loss
        print("diff min max", min_before, max_before)
        if num_trajectories > 0:
            total_loss /= num_trajectories

        return total_loss

    def update(self, samples: Dict, epoch_cur) -> Dict:
        """
        执行PPO更新
        """
        trajectories = samples['trajectories']
        rewards = samples['rewards']
        conditions = samples['conditions']

        # 将奖励转换为 Tensor
        rewards = torch.stack(rewards).to(self.device)
        generated_motions = torch.stack(samples['generated_motions']).to(self.device)
        with torch.no_grad():
            values = self.value_head(torch.stack(samples['conditions']).to(self.device)).squeeze(-1)

            advantages = self.compute_advantages(rewards, values)

            # returns = advantages + values.squeeze(-1) ###
            returns = rewards.detach()
            advantages = advantages.detach()
            # returns = returns.detach()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(trajectories))

            for start in range(0, len(trajectories), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_trajectories = [trajectories[i] for i in batch_indices]
                batch_conditions = torch.stack([conditions[i] for i in batch_indices])
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                new_log_probs_list = []
                old_log_probs_list = []

                for i, traj in enumerate(batch_trajectories):
                    new_log_probs = self.policy.compute_new_log_prob(
                        traj, batch_conditions[i], self.env.cur_extra_info
                    )
                    new_log_probs_list.append(new_log_probs)

                    old_log_probs = samples['log_probs'][batch_indices[i]]
                    old_log_probs_list.append(old_log_probs)

                policy_loss = self.compute_ppo_loss(
                    old_log_probs_list, new_log_probs_list, batch_advantages
                )

                generated = torch.stack([
                    samples['generated_motions'][i] for i in batch_indices
                ])
                pred_values = self.value_head(batch_conditions).squeeze(-1) ### generated
                value_loss = F.mse_loss(pred_values, batch_returns)

                entropy = self._compute_entropy(new_log_probs_list)

                total_loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                # print(self.value_head.layer.weight.grad)
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.policy.get_parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)

                # 检查 Value Head 是否有梯度
                # for name, param in self.value_head.named_parameters():
                #     if param.grad is None:
                #         print(f"[警告] Value Head 参数 {name} 没有梯度！")
                #     else:
                #         grad_norm = param.grad.norm().item()
                #         print(f"[调试] Value Head {name} 梯度模长: {grad_norm:.8f}")
                #
                # # 检查 策略网络 (BMDM) 是否有梯度
                # for name, param in self.policy.bmdm.named_parameters():
                #     if param.grad is not None:
                #         if param.grad.norm() > 0:
                #             print(f"[调试] 策略网络有梯度更新")
                #             break  # 只要有一个有就行
                #     # else:
                #     #     print(f"[调试] 策略网络没有梯度更新")
                # old_weight = self.value_head[0].weight.data.clone()
                if epoch_cur >= 20:
                    self.optimizer.step()
                self.value_optimizer.step()

                # weight_diff = torch.norm(self.value_head[0].weight.data - old_weight)
                # print(f"Value Head 权重变化量: {weight_diff.item():.10f}")

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy

            with torch.no_grad():
                approx_kl_list = []
                for i, traj in enumerate(batch_trajectories):
                    old_lp = torch.stack(samples['log_probs'][batch_indices[i]]).to(self.device)
                    new_lp = torch.stack(new_log_probs_list[i]).detach()
                    # 计算近似 KL 散度
                    kl = (old_lp - new_lp).mean().item()
                    approx_kl_list.append(kl)

                mean_kl = abs(sum(approx_kl_list) / len(approx_kl_list))

                # 如果单轮 epoch 更新导致策略偏移过大，立刻终止当前 Batch 的剩余 PPO epochs
                if mean_kl > 0.05:
                    print(f"  [保护机制触发] Epoch {epoch}: 近似 KL ({mean_kl:.4f}) 超出阈值，触发早停！")
                    break  # 跳出 ppo_epochs 循环

        import math
        actual_batches_per_epoch = math.ceil(len(trajectories) / self.mini_batch_size)
        num_updates = self.ppo_epochs * actual_batches_per_epoch

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def _compute_entropy(self, log_probs_list: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        计算熵
        """
        total_entropy = 0
        count = 0

        for log_probs in log_probs_list:
            for lp in log_probs:
                if lp is not None:
                    entropy = -lp.exp() * lp
                    total_entropy += entropy.mean()
                    count += 1

        # return total_entropy / max(count, 1)
        return torch.tensor(0.0, device=self.device)

    def train_controller(self, out_model_file: str, int_output_dir: str):
        """
        主训练循环
        """
        # obs = self.env.reset()

        num_epochs = self.ddpo_config.get('num_epochs', 10000)

        for epoch in range(num_epochs):
            samples = self.collect_samples(self.num_samples_per_epoch)

            loss_dict = self.update(samples, epoch)

            self.update_count += 1

            if self.lr_decay_type == 'linear':
                lr = self.lr - (self.lr - self.final_lr) * epoch / num_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif self.lr_decay_type == 'exponential':
                lr = self.lr * (0.99 ** epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            # print("foot_slide", samples['reward_dict'][0])
            stats = {
                'epoch': epoch,
                'policy_loss': loss_dict['policy_loss'],
                'value_loss': loss_dict['value_loss'],
                'entropy': loss_dict['entropy'],
                'mean_reward': torch.stack(samples['rewards']).mean().item(),
                'foot_slide': samples['reward_dict'][0]['foot_slide'].mean().item(),
                'floating': samples['reward_dict'][0]['floating'].mean().item(),
                'ground_pen': samples['reward_dict'][0]['ground_penetration'].mean().item(),
                'smoothness': samples['reward_dict'][0]['smoothness'].mean().item(),
            }

            if hasattr(self, 'logger'):
                self.logger.log_epoch(stats, step=epoch)
                self.logger.print_log(stats)

            if epoch % self.save_interval == 0:
                save_util.save_weight(copy.deepcopy(self.bmdm), out_model_file)
                save_util.save_weight(copy.deepcopy(self.bmdm), f'{int_output_dir}/_ep{epoch}.pth')
                print(f"Model saved at epoch {epoch}")

        save_util.save_weight(copy.deepcopy(self.bmdm), out_model_file)
        print("Training completed!")

    def test_controller(self):
        """
        测试控制器
        """
        self.env.reset_initial_frames()

        num_test_episodes = self.ddpo_config.get('num_test_episodes', 10)
        total_reward = 0

        for episode in range(num_test_episodes):
            condition = self.env.get_cond_frame()

            with torch.no_grad():
                generated_motion, _ = self.policy(condition, self.env.cur_extra_info)

            reward, reward_dict = self.reward_calculator.compute_reward(
                generated_motion, condition, self.prev_motion
            )

            total_reward += reward.mean().item()
            self.prev_motion = generated_motion

            print(f"Episode {episode + 1}: Reward = {reward.mean().item():.4f}")

        avg_reward = total_reward / num_test_episodes
        print(f"Average Reward over {num_test_episodes} episodes: {avg_reward:.4f}")

        return avg_reward
