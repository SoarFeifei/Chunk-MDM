import torch
import torch.nn as nn
import numpy as np


class PhysicsConstraintReward(nn.Module):
    """
    物理约束奖励系统
    包含滑步检测、悬空检测、穿地检测等物理约束奖励/惩罚机制
    """
    NAME = 'PhysicsConstraint'

    def __init__(self, config, dataset, device):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = device

        # 物理约束参数
        self.physics_config = config.get('physics_constraints', {})
        self.foot_slide_weight = self.physics_config.get('foot_slide_weight', 10.0)
        self.floating_weight = self.physics_config.get('floating_weight', 10.0)
        self.ground_penetration_weight = self.physics_config.get('ground_penetration_weight', 10.0)
        self.smoothness_weight = self.physics_config.get('smoothness_weight', 5)

        # 阈值参数
        self.contact_threshold = self.physics_config.get('contact_threshold', 0.05)
        self.slide_threshold = self.physics_config.get('slide_threshold', 0.02)
        self.ground_threshold = self.physics_config.get('ground_threshold', 0.01)

        # 块大小配置
        self.block_size = config['block_size']
        self.frame_dim = dataset.frame_dim
        self.single_frame_dim = int(self.frame_dim / self.block_size)

        # 脚部关节索引
        self.foot_idx = dataset.foot_idx
        self.toe_idx = dataset.toe_idx if hasattr(dataset, 'toe_idx') else []

        # 历史记录
        self.foot_pos_history = None
        self.history_size = 2

    def reset_history(self, batch_size):
        """重置历史记录"""
        if self.foot_pos_history is None or self.foot_pos_history.shape[0] != batch_size:
            num_feet = len(self.foot_idx) + len(self.toe_idx)
            self.foot_pos_history = torch.zeros(
                (batch_size, self.history_size, num_feet * 3),
                device=self.device
            )

    def _extract_foot_positions(self, pose):
        """
        从姿态中提取脚部关节位置
        pose: [batch_size, block_size * single_frame_dim]
        return: [batch_size, block_size, num_feet * 3]
        """
        batch_size = pose.shape[0]
        foot_positions = []
        # print("pose", pose.shape)

        for i in range(self.block_size):
            frame_pose = pose[:, i * self.single_frame_dim:(i + 1) * self.single_frame_dim]
            pose_denorm = self.dataset.denorm_data(frame_pose, device=self.device)

            # 获取关节位置
            jnts = self.dataset.jnts_frame_pt(pose_denorm)

            # 提取脚部位置
            foot_pos = []
            for idx in self.foot_idx:
                foot_pos.append(jnts[:, idx, :])    #
                # print("idx max min", idx, jnts[:, idx, :])
            for idx in self.toe_idx:
                foot_pos.append(jnts[:, idx, :])
                # print("idx max min", idx, jnts[:, idx, :])

            foot_pos = torch.cat(foot_pos, dim=-1)
            foot_positions.append(foot_pos)

        foot_positions = torch.stack(foot_positions, dim=1)
        # print("foot_positions", foot_positions.shape)

        return foot_positions

    def calc_foot_slide(self, pose):
        """
        计算滑步惩罚
        pose: [batch_size, block_size * single_frame_dim]
        return: [batch_size, 1]
        """
        batch_size = pose.shape[0]

        if self.foot_pos_history is None:
            self.reset_history(batch_size)

        # 获取当前块的脚部位置
        current_foot_pos = self._extract_foot_positions(pose)

        slide_penalty = torch.zeros(batch_size, 1, device=self.device)

        for i in range(self.block_size):
            if i == 0:
                # 使用历史记录
                prev_pos = self.foot_pos_history[:, -1, :]
            else:
                prev_pos = current_foot_pos[:, i - 1, :]

            curr_pos = current_foot_pos[:, i, :]
            # print("curr_pos", curr_pos)
            # 计算位移
            displacement = torch.norm(curr_pos - prev_pos, dim=-1, keepdim=True)

            # 检查是否接触地面
            foot_heights = curr_pos.view(batch_size, -1, 3)[:, :, 1]
            in_contact = foot_heights < self.contact_threshold
            contact_coef = in_contact.float().mean(dim=-1, keepdim=True)

            # 滑步惩罚：接触时位移过大
            slide = contact_coef * torch.clamp(displacement - self.slide_threshold, min=0)
            slide_penalty += slide

        # 更新历史记录
        self.foot_pos_history = torch.roll(self.foot_pos_history, shifts=-1, dims=1)
        self.foot_pos_history[:, -1, :] = current_foot_pos[:, -1, :]
        # print("slide_penalty", slide_penalty.shape)

        return slide_penalty / self.block_size

    def calc_floating(self, pose):
        """
        计算悬空惩罚
        pose: [batch_size, block_size * single_frame_dim]
        return: [batch_size, 1]
        """
        batch_size = pose.shape[0]
        floating_penalty = torch.zeros(batch_size, 1, device=self.device)

        for i in range(self.block_size):
            frame_pose = pose[:, i * self.single_frame_dim:(i + 1) * self.single_frame_dim]
            pose_denorm = self.dataset.denorm_data(frame_pose, device=self.device)
            jnts = self.dataset.jnts_frame_pt(pose_denorm)

            # 获取脚部高度
            foot_heights = []
            for idx in self.foot_idx:
                foot_heights.append(jnts[:, idx, 1:2])
            for idx in self.toe_idx:
                foot_heights.append(jnts[:, idx, 1:2])

            foot_heights = torch.cat(foot_heights, dim=-1)

            # 检查是否所有脚都悬空
            all_floating = (foot_heights > self.contact_threshold).all(dim=-1, keepdim=True)
            floating_penalty += all_floating.float()

        return floating_penalty / self.block_size

    def calc_ground_penetration(self, pose):
        """
        计算穿地惩罚
        pose: [batch_size, block_size * single_frame_dim]
        return: [batch_size, 1]
        """
        batch_size = pose.shape[0]
        penetration_penalty = torch.zeros(batch_size, 1, device=self.device)

        for i in range(self.block_size):
            frame_pose = pose[:, i * self.single_frame_dim:(i + 1) * self.single_frame_dim]
            pose_denorm = self.dataset.denorm_data(frame_pose, device=self.device)
            jnts = self.dataset.jnts_frame_pt(pose_denorm)

            # 获取所有关节高度
            joint_heights = jnts[:, :, 1]

            # 穿地惩罚：关节高度低于地面阈值
            penetration = torch.clamp(self.ground_threshold - joint_heights, min=0)
            penetration_penalty += penetration.mean(dim=-1, keepdim=True)

        return penetration_penalty * 100 / self.block_size

    def calc_smoothness(self, pose, prev_pose=None):
        """
        计算运动平滑度奖励
        pose: [batch_size, block_size * single_frame_dim]
        prev_pose: [batch_size, block_size * single_frame_dim], optional
        return: [batch_size, 1]
        """
        batch_size = pose.shape[0]
        smoothness_loss = torch.zeros(batch_size, 1, device=self.device)

        # 块内平滑度
        for i in range(1, self.block_size):
            curr_frame = pose[:, i * self.single_frame_dim:(i + 1) * self.single_frame_dim]
            prev_frame = pose[:, (i - 1) * self.single_frame_dim:i * self.single_frame_dim]
            smoothness_loss += torch.norm(curr_frame - prev_frame, dim=-1, keepdim=True)

        # 块间平滑度
        if prev_pose is not None:
            last_frame_prev = prev_pose[:, -self.single_frame_dim:]
            first_frame_curr = pose[:, :self.single_frame_dim]
            smoothness_loss += torch.norm(first_frame_curr - last_frame_prev, dim=-1, keepdim=True)

        # 平滑度越高（速度变化小），奖励越高
        smoothness_reward = -smoothness_loss
        return smoothness_reward / (self.block_size if prev_pose is None else self.block_size + 1)

    def compute_total_reward(self, pose, prev_pose=None):
        """
        计算总的物理约束奖励
        pose: [batch_size, block_size * single_frame_dim]
        prev_pose: [batch_size, block_size * single_frame_dim], optional
        return: dict containing all reward components
        """
        rewards = {}

        # 滑步惩罚（负值）
        foot_slide = self.calc_foot_slide(pose)
        rewards['foot_slide'] = -self.foot_slide_weight * foot_slide

        # 悬空惩罚（负值）
        floating = self.calc_floating(pose)
        rewards['floating'] = -self.floating_weight * floating

        # 穿地惩罚（负值）
        ground_penetration = self.calc_ground_penetration(pose)
        rewards['ground_penetration'] = -self.ground_penetration_weight * ground_penetration

        # 平滑度奖励
        smoothness = self.calc_smoothness(pose, prev_pose)
        rewards['smoothness'] = self.smoothness_weight * smoothness

        # 总奖励
        rewards['total_physics'] = sum(rewards.values())

        return rewards

    def forward(self, pose, prev_pose=None):
        """
        前向传播，返回总物理约束奖励
        """
        rewards = self.compute_total_reward(pose, prev_pose)
        return rewards['total_physics'], rewards
