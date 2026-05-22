# Portions based in part of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

# Copyright (c) 2017 Ilya Kostrikov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.optim as optim
from policy.learning.storage import RolloutStorage
from policy.common.misc_utils import update_exponential_schedule, update_linear_schedule
import util.logging as logging_util
import util.save as save_util
import copy
import yaml
from policy.common.misc_utils import EpisodeRunner
import util.eval as eval_util
from dataset.util.skeleton_info import LAFAN1_links
from dataset.util.skeleton_info import SMPL_links
import numpy as np

class PPOAgent(object):
    NAME = 'PPO'
    def __init__(self, config, actor_critic, env, device):
        self.mirror_function = None
        self.config = config
        self.device = device
        self.env = env
        self.actor_critic = actor_critic.to(self.device)
        
        self.num_parallel = env.num_parallel
        self.mini_batch_size = config["mini_batch_size"]

        num_frames = 10e9
        self.num_steps_per_rollout = self.env.max_timestep
        self.num_updates = int( num_frames / self.num_parallel / self.num_steps_per_rollout)
        self.num_mini_batch = int( self.num_parallel * self.num_steps_per_rollout / self.mini_batch_size)
        self.num_epoch = 0
        obs_shape = self.env.observation_space.shape    # (1337,)
        # print(obs_shape)
        obs_shape = (obs_shape[0], *obs_shape[1:])  # (1337,)
        # print(self.env.observation_space.shape, obs_shape)

        self.rollouts = RolloutStorage(
            self.num_steps_per_rollout,
            self.num_parallel,
            obs_shape,
            self.actor_critic.actor.action_dim,
            self.actor_critic.state_size,
        )
        # print(self.num_steps_per_rollout,
        #     self.num_parallel,
        #     obs_shape,
        #     self.actor_critic.actor.action_dim,
        #     self.actor_critic.state_size,)
        
        self.use_gae = config["use_gae"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]

        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.lr = config["lr"]
        self.final_lr = config["final_lr"]
        self.lr_decay_type = config["lr_decay_type"]
        self.eps = config["eps"]
        self.save_interval = config["save_interval"]

        ###
        self.per_frame_dim = int(self.env.frame_dim / self.env.block_size)
        self.lastblock = torch.zeros(
            (self.num_parallel, self.per_frame_dim * (self.env.block_size - 1)))

        self.action_steps = self.env.config['action_step']
        self.action_rgr_steps = [self.action_steps.index(s)+1 for s in self.env.config.get('action_rgr_step', [])]
        self.action_mask = torch.zeros(self.mini_batch_size, len(self.action_steps)+1, self.env.frame_dim).to(self.device)  ### per frame
        # self.action_mask = torch.ones(self.mini_batch_size, self.env.block_size, 3).to(self.device)    ###
        if len(self.action_rgr_steps) > 0:        ###
            self.action_mask[:, self.action_rgr_steps] = 1
        self.action_mask = self.action_mask.view(self.mini_batch_size,-1)
        self.actor_reg_weight = config.get('actor_reg_weight',1)
        self.actor_bound_weight = config.get('actor_bound_weight',0.0)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)
        if not self.env.is_rendered:
            print(env.int_output_dir, "HCONTROL_{}_{}_{}_{}".format(env.NAME, env.model.NAME, env.dataset.NAME, self.NAME))
            self.logger = logging_util.wandbLogger(run_name=env.int_output_dir, proj_name="HCONTROL_{}_{}_{}_{}".format(env.NAME, env.model.NAME, env.dataset.NAME, self.NAME))

    def test_controller(self):
        self.num_parallel = self.env.num_parallel_test
        obs = self.env.reset()
        ep_reward = 0

        # self.env.reset_initial_frames()
        # print("obs", obs.shape)
        foot_slide = bone_err = 0
        pen_freq = 0
        pen_dist = 0
        test_num = 20
        if self.env.dataset.dataset_name == "LAFAN1":
            foot_idx = [3, 4, 7, 8]
            links = LAFAN1_links
        elif self.env.dataset.dataset_name == "STYLE100":
            links = LAFAN1_links
            foot_idx = [17, 18, 21, 22]
        elif self.env.dataset.dataset_name == "AMASS":
            links = SMPL_links
            foot_idx = [7, 8, 10, 11]
        # ref_clip = obs[0].reshape(self.env.dataset.block_size,
        #                             int(self.env.dataset.frame_dim / self.env.dataset.block_size)).detach().cpu().numpy()
        ref_clip = self.env.dataset.motion_flattened[0:int(60 / self.env.dataset.block_size) + 1]
        ref_clip = ref_clip.reshape(ref_clip.shape[0] * self.env.dataset.block_size, int(self.env.dataset.frame_dim / self.env.dataset.block_size))
        denorm_ref_clip = self.env.dataset.denorm_data(ref_clip)
        ref_clip = self.env.dataset.x_to_jnts(denorm_ref_clip, mode=self.env.dataset.data_component[0])[None, ...]
        sk_length = eval_util.extract_sk_lengths(LAFAN1_links, ref_clip)

        ### test
        for j in range(test_num):
            obs = self.env.reset()
            obs_seq = obs
            obs_seq = obs_seq.unsqueeze(0).view(self.num_parallel, 1, -1)
            obs_jnts_list = []
            for i in range(int(60 / self.env.dataset.block_size)):
                with torch.no_grad():
                    action = self.actor_critic.actor(obs)
                obs, reward, done, info = self.env.step(action)
                obs_temp = obs
                obs_seq = torch.cat([obs_seq, obs_temp.unsqueeze(0).view(self.num_parallel, 1, -1)], dim=1)
                # print("obs_seq", obs_seq.shape)
            obs_seq = obs_seq.view(self.num_parallel, -1, self.per_frame_dim).detach().cpu().numpy()
            # print("obs_seq", obs_seq.shape)

            for i in range(self.num_parallel):
                pred_long = obs_seq[i, :, :]  # .detach().cpu().numpy()
                denorm_pred_long = self.env.dataset.denorm_data(copy.deepcopy(pred_long))
                pred_long_jnts = self.env.dataset.x_to_jnts(denorm_pred_long, mode=self.env.dataset.data_component[0])[
                    None, ...]
                pred_long_jnts = pred_long_jnts.squeeze(0)
                for mode in self.env.dataset.data_component:
                    jnts_mode = self.env.dataset.x_to_jnts(denorm_pred_long, mode=mode)
                    if mode == self.env.dataset.data_component[0]:
                        obs_jnts_list.append(jnts_mode)

                sk_length_pred = eval_util.extract_sk_lengths(links, pred_long_jnts)
                bone_err_per_frame = np.abs(sk_length_pred - sk_length)  # [21,125]
                bone_err += bone_err_per_frame.mean() / (test_num * self.num_parallel)
                contact_zs_mean, contact_event = eval_util.compute_ground_pen(foot_idx, pred_long_jnts, -0.03)
                pen_freq += contact_event / (test_num * self.num_parallel)
                pen_dist += contact_zs_mean / (test_num * self.num_parallel)
            # print("obs_jnts_list", obs_jnts_list.shape)
            obs_jnts_list = np.array(obs_jnts_list)
            foot_slide += eval_util.compute_foot_slide(foot_idx, obs_jnts_list)
            print("foot_slide", foot_slide * 100 / test_num)
            print("pen_freq", pen_freq * 100)
            print("pen_dist", pen_dist * 100)
            print("bone_err", bone_err)

        # with EpisodeRunner(self.env) as runner:
        #
        #     while not runner.done:
        #
        #         all_next = []  ###
        #         with torch.no_grad():
        #             obs_first = torch.cat([obs[:, :self.per_frame_dim], obs[:, -2:]], dim=1)
        #             action = self.actor_critic.actor(obs)     ### per frame obs[:, -(int(self.env.frame_dim / self.env.block_size) + 2):]
        #         # for i in range(20):   ###
        #         #     obs, reward, done, info = self.env.step(action)  # 或改为 model.sample_rl_ddpm(obs_t, action_dict, extra_info)
        #         # # obs_, reward_, done_, info_ = self.env.step(-action)
        #         #     obs_first = torch.cat([obs[:, :self.per_frame_dim], obs[:, -2:]], dim=1)
        #         #     all_next.append(obs_first.detach().cpu())
        #
        #         # diff = (obs - obs_).norm(dim=-1).mean()
        #         # print("Mean effect of flipping action:", diff)
        #         # all_next = torch.stack(all_next, dim=0)  # [n_repeat, B, frame_dim]
        #         # mean_next = all_next.mean(0)
        #         # std_next = all_next.std(0)
        #         # # 计算每个样本的平均标准差（表示扩散噪声强度）
        #         # avg_std = std_next.mean().item()
        #         # print(f"Average per-dim std of next_obs: {avg_std:.6f}")
        #         # # 计算相同action多次生成的pairwise差异
        #         # diffs = (all_next[1:] - all_next[:-1]).norm(dim=-1).mean().item()
        #         # print(f"Average pairwise difference between samples: {diffs:.6f}")
        #         # print("mean_next", mean_next)
        #
        #         # print("action", action.shape)   ### action torch.Size([1, 3738])
        #         obs, reward, done, info = self.env.step(action)
        #         # print("obs", obs.shape)   ### obs torch.Size([1, 269]) condition + delta torch.Size([1, 267]) torch.Size([1, 2])
        #         ep_reward += reward
        #
        #         if done.any():
        #             print("--- Episode reward: %2.4f" % float(ep_reward[done].mean()))
        #             ep_reward *= (~done).float()
        #             reset_indices = self.env.parallel_ind_buf.masked_select(done.squeeze())
        #             obs = self.env.reset_index(reset_indices)
        #
        #         if info.get("reset"):
        #             print("--- Episode reward: %2.4f" % float(ep_reward.mean()))
        #             ep_reward = 0
        #             obs = self.env.reset()


    def compute_action_bound_loss(self, norm_a, bound_min=-1, bound_max=1):
        violation_min = torch.clamp_max(norm_a.mean() - bound_min, 0.0)
        violation_max = torch.clamp_min(norm_a.mean() - bound_max, 0)
        bound_violation_loss = torch.sum(torch.square(violation_min), dim=-1) \
                    + torch.sum(torch.square(violation_max), dim=-1)
        return bound_violation_loss.mean()

    def compute_action_reg_weight(self, norm_a, mask):
        # print("norm_a, mask", norm_a.shape, mask.shape)
        norm_a = norm_a * mask
        action_reg_loss = torch.sum(torch.square(norm_a), dim=-1)
        return action_reg_loss.mean()


    def train_controller(self, out_model_file, int_output_dir):
        obs = self.env.reset()
        # print("obs", obs.shape, self.rollouts.observations.shape)
        # self.lastblock = obs[:, :self.per_frame_dim * (self.env.block_size - 1)]
        # obs_first = torch.cat([obs[:, :self.per_frame_dim], obs[:, -2:]], dim=1)
        # obs_mean = obs[:, :self.env.frame_dim].reshape(obs.shape[0], self.env.block_size, -1).mean(dim=1)
        # obs_mean = torch.cat([obs_mean.reshape(obs.shape[0], -1), obs[:, -2:]], dim=1)

        # self.rollouts.observations = self.rollouts.observations.to(self.device)
        print("self.rollouts.observations", self.rollouts.observations.shape, obs.shape)
        self.rollouts.observations[0].copy_(obs)  ###obs[:, -(self.per_frame_dim + 2):]
        # print("obs", obs.shape, self.rollouts.observations[0].shape)
        self.rollouts.to(self.device)
        num_samples = 0
        # print("num_updates:", self.num_updates)  ### 65104
        for update in range(self.num_updates):

            ep_info = {"reward": []}
            ep_reward = 0

            if self.lr_decay_type == "linear":
                update_linear_schedule(
                    self.optimizer, update, self.num_updates, self.lr, self.final_lr
                )
            elif self.lr_decay_type == "exponential":
                update_exponential_schedule(
                    self.optimizer, update, 0.99, self.lr, self.final_lr
                )

            for step in range(self.num_steps_per_rollout):  # 在每个更新回合中，采样固定步数 self.num_steps_per_rollout 300
                # Sample actions
                # print("step:",step)
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(
                        self.rollouts.observations[step]   ### per frame
                    )
                # print("action max min", action.max().item(), action.min().item())
                # print("action:", action.shape)  ### action: torch.Size([512, 3738])
                # print("self.rollouts.observations:", self.rollouts.observations.shape)  ### torch.Size([301, 512, 269])

                ### qc step
                # for i in len(self.action_steps + 1):
                #     obs, reward, done, info = self.env.step(action[..., i*self.env.frame_dim:(i+1)*self.env.frame_dim])

                obs, reward, done, info = self.env.step(action)
                # print("obs:", obs.shape)  ### obs: torch.Size([512, 269])
                ep_reward += reward

                end_of_rollout = info.get("reset")
                
                
                masks = (~done).float()
                bad_masks = (~(done * end_of_rollout)).float()

                if done.any():
                    ep_info["reward"].append(ep_reward[done].clone())
                    ep_reward *= (~done).float()  # zero out the dones
                    reset_indices = self.env.parallel_ind_buf.masked_select(done.squeeze())
                    obs = self.env.reset_index(reset_indices)

                if torch.all(end_of_rollout):
                    obs = self.env.reset()

                # obs_first = torch.cat([obs[:, :self.per_frame_dim], obs[:, -2:]], dim=1)
                # obs_mean = obs[:, :self.env.frame_dim].reshape(obs.shape[0], self.env.block_size, -1).mean(dim=1)
                # obs_mean = torch.cat([obs_mean.reshape(obs.shape[0], -1), obs[:, -2:]], dim=1)
                self.rollouts.insert(
                    obs, action, action_log_prob, value, reward, masks, bad_masks   # 将这一步的 observation（obs）添加到 self.rollouts.observations[step + 1]。
                )   ### per frame   obs[:, -(self.per_frame_dim + 2):]
                
            num_samples += (obs.shape[0]*self.num_steps_per_rollout)
            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.rollouts.observations[-1]).detach()  ### per frame

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda)

            value_loss, action_loss, dist_entropy, regr = self.update(self.rollouts)

            self.rollouts.after_update()

            save_util.save_weight(copy.deepcopy(self.actor_critic), out_model_file)
            if update % self.save_interval == 0:
                save_util.save_weight(copy.deepcopy(self.actor_critic), int_output_dir + '/_ep{}.pth'.format(update))

            ep_info["reward"] = torch.cat(ep_info["reward"])
            
            stats = {
                    "update": update,
                    "reward_mean": torch.mean(ep_info['reward']),
                    "reward_max": torch.max(ep_info['reward']),
                    "reward_min": torch.min(ep_info['reward']),
                    "dist_entropy": dist_entropy,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "regr": regr, 
                }
            self.logger.log_epoch(stats, step=int(num_samples))
            self.logger.print_log(stats)
    


    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        print(f"[探针2] 优势函数 Advantage 均值: {advantages.mean().item():.4f}, "
              f"标准差: {advantages.std().item():.4f}, 极值: [{advantages.min().item():.2f}, {advantages.max().item():.2f}]")
        invalid_mask = torch.isnan(advantages) | torch.isinf(advantages)

        # 2. 揪出极其离谱的极端值 (比如 Advantage 超过 1000 或低于 -1000)
        extreme_mask = advantages.abs() > 1000.0

        # 3. 将这些“坏苹果”的优势函数强行置零 (让 Actor 在这一步不更新它们)
        bad_apple_mask = invalid_mask | extreme_mask
        advantages[bad_apple_mask] = 0.0

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                if self.mirror_function is not None:
                    (
                        observations_batch,
                        actions_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = self.mirror_function(sample)
                else:
                    (
                        observations_batch,
                        actions_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = sample

                #print(actions_batch.shape, observations_batch.shape)
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    observations_batch, actions_batch     ### per frame [:, -(int(self.env.frame_dim / self.env.block_size) + 2):]
                )
                # print("actions_batch", actions_batch.shape)
                # print(f"[探针1] 动作均值 绝对值均值: {actions_batch.abs().mean().item():.4f}, "
                #       f"最大值: {actions_batch.max().item():.4f}, 最小值: {actions_batch.min().item():.4f}")
                #print(action_log_probs, old_action_log_probs_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )

                #$$
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_param).float()).item()
                # print(f"[探针3] PPO 截断率: {clip_fraction * 100:.1f}% | Ratio 均值: {ratio.mean().item():.2f}")

                # print("ratio", ratio.mean(), ratio.max().item(), ratio.min().item())
                # print("action_log_probs", action_log_probs.mean(), action_log_probs.max().item(), action_log_probs.min().item())
                # print("old_action_log_probs_batch", old_action_log_probs_batch.mean(), old_action_log_probs_batch.max().item(), old_action_log_probs_batch.min().item())
                action_loss = -torch.min(surr1, surr2).mean()
                regr = 0.0
                if self.actor_reg_weight > 0.0:
                    action_rgr_loss = self.actor_reg_weight * self.compute_action_reg_weight(actions_batch, self.action_mask)
                   
                    action_loss += action_rgr_loss
                    regr += action_rgr_loss
                
                if self.actor_bound_weight > 0.0:
                    action_bound_loss = self.actor_bound_weight * self.compute_action_bound_loss(actions_batch)
                    action_loss += action_bound_loss
                    regr += action_bound_loss
                
                
                value_loss = (return_batch - values).pow(2).mean()
                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, regr
