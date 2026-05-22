import torch
import torch.nn as nn
import copy
import numpy as np


class BMDMExplorer(nn.Module):
    """
    块级BMDM探索模块
    用于提供探索性动作样本和样板动作
    """
    NAME = 'BMDMExplorer'

    def __init__(self, config, bmdm_model, dataset, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dataset = dataset
        self.block_size = config['block_size']

        # 冻结BMDM模型
        self.bmdm_model = copy.deepcopy(bmdm_model)
        self.bmdm_model.eval()
        for param in self.bmdm_model.parameters():
            param.requires_grad = False

        # 探索配置
        self.explorer_config = config.get('bmdm_explorer', {})
        self.num_exploration_samples = self.explorer_config.get('num_exploration_samples', 5)
        self.exploration_noise_scale = self.explorer_config.get('exploration_noise_scale', 0.1)
        self.use_best_of_n = self.explorer_config.get('use_best_of_n', True)

        # 样板动作缓存
        self._template_cache = {}

    def generate_exploration_samples(self, current_state, extra_info=None, num_samples=None):
        """
        生成探索性动作样本
        current_state: [batch_size, state_dim]
        return: [batch_size, num_samples, action_dim]
        """
        if num_samples is None:
            num_samples = self.num_exploration_samples

        batch_size = current_state.shape[0]
        action_dim = current_state.shape[1]

        with torch.no_grad():
            exploration_samples = []

            for _ in range(num_samples):
                # 添加随机噪声进行探索
                noise = torch.randn_like(current_state) * self.exploration_noise_scale
                noisy_state = current_state + noise

                # 通过BMDM生成动作
                sample = self.bmdm_model.eval_step(noisy_state, extra_info)
                exploration_samples.append(sample)

            exploration_samples = torch.stack(exploration_samples, dim=1)

        return exploration_samples

    def generate_template_action(self, current_state, extra_info=None):
        """
        生成样板动作
        current_state: [batch_size, state_dim]
        return: [batch_size, action_dim], reward_estimate
        """
        with torch.no_grad():
            if self.use_best_of_n:
                # 生成多个样本，选择最好的
                samples = self.generate_exploration_samples(
                    current_state, extra_info, num_samples=max(self.num_exploration_samples, 3)
                )

                # 简单的启发式评分：选择最接近原状态的（或可以用其他评分）
                scores = -torch.norm(samples - current_state.unsqueeze(1), dim=-1)
                best_idx = torch.argmax(scores, dim=1)

                batch_indices = torch.arange(current_state.shape[0], device=self.device)
                best_sample = samples[batch_indices, best_idx]
                best_score = scores[batch_indices, best_idx]

                return best_sample, best_score
            else:
                # 直接生成一个样板
                template = self.bmdm_model.eval_step(current_state, extra_info)
                return template, torch.zeros(current_state.shape[0], device=self.device)

    def get_template_from_dataset(self, batch_size, template_type='random'):
        """
        从数据集中获取样板动作
        template_type: 'random', 'walking', 'running' 等
        return: [batch_size, action_dim]
        """
        cache_key = (batch_size, template_type)

        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        # 从数据集中随机选择
        indices = np.random.choice(len(self.dataset.valid_idx), batch_size, replace=False)
        start_indices = self.dataset.valid_idx[indices]

        templates = torch.tensor(
            self.dataset.motion_flattened[start_indices],
            device=self.device,
            dtype=torch.float32
        )

        self._template_cache[cache_key] = templates
        return templates

    def forward(self, current_state, extra_info=None, mode='explore'):
        """
        前向传播
        mode: 'explore' - 生成探索样本
              'template' - 生成样板动作
              'dataset_template' - 从数据集获取样板
        """
        if mode == 'explore':
            return self.generate_exploration_samples(current_state, extra_info)
        elif mode == 'template':
            return self.generate_template_action(current_state, extra_info)
        elif mode == 'dataset_template':
            return self.get_template_from_dataset(current_state.shape[0]), None
        else:
            raise ValueError(f"Unknown mode: {mode}")


class BMDMGuide(nn.Module):
    """
    BMDM引导模块
    用于评估和引导PPO策略的动作
    """
    NAME = 'BMDMGuide'

    def __init__(self, config, bmdm_model, dataset, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dataset = dataset
        self.block_size = config['block_size']

        # 冻结的BMDM模型
        self.bmdm_model = copy.deepcopy(bmdm_model)
        self.bmdm_model.eval()
        for param in self.bmdm_model.parameters():
            param.requires_grad = False

        # 引导配置
        self.guide_config = config.get('bmdm_guide', {})
        self.guide_weight = self.guide_config.get('guide_weight', 0.5)
        self.similarity_threshold = self.guide_config.get('similarity_threshold', 0.8)

    def compute_action_similarity(self, ppo_action, bmdm_action):
        """
        计算PPO动作和BMDM动作的相似度
        return: [batch_size]
        """
        # 余弦相似度
        ppo_action_norm = torch.nn.functional.normalize(ppo_action, dim=-1)
        bmdm_action_norm = torch.nn.functional.normalize(bmdm_action, dim=-1)
        similarity = (ppo_action_norm * bmdm_action_norm).sum(dim=-1)
        return similarity

    def compute_guide_reward(self, ppo_action, current_state, extra_info=None):
        """
        计算引导奖励
        return: [batch_size, 1], similarity_scores
        """
        with torch.no_grad():
            # 获取BMDM的推荐动作
            bmdm_action = self.bmdm_model.eval_step(current_state, extra_info)

            # 计算相似度
            similarity = self.compute_action_similarity(ppo_action, bmdm_action)

            # 引导奖励：相似度高则奖励
            guide_reward = self.guide_weight * similarity.unsqueeze(1)

            return guide_reward, similarity

    def should_guide(self, similarity):
        """
        判断是否需要引导
        """
        return similarity < self.similarity_threshold

    def forward(self, ppo_action, current_state, extra_info=None):
        """
        前向传播，返回引导奖励和相似度
        """
        return self.compute_guide_reward(ppo_action, current_state, extra_info)
