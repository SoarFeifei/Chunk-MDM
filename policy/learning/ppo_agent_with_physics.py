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
from policy.learning.physics_constraints import PhysicsConstraintReward
from policy.learning.bmdm_integration import BMDMExplorer, BMDMGuide


class PPOAgentWithPhysics(object):
    NAME = 'PPOWithPhysics'

    def __init__(self, config, actor_critic, env, device, bmdm_model=None):
        self.mirror_function = None
        self.config = config
        self.device = device
        self.env = env
        self.actor_critic = actor_critic.to(self.device)

        self.num_parallel = env.num_parallel
        self.mini_batch_size = config["mini_batch_size"]

        num_frames = 10e9
        self.num_steps_per_rollout = self.env.max_timestep
        self.num_updates = int(num_frames / self.num_parallel / self.num_steps_per_rollout)
        self.num_mini_batch = int(self.num_parallel * self.num_steps_per_rollout / self.mini_batch_size)
        self.num_epoch = 0
        obs_shape = self.env.observation_space.shape
        obs_shape = (obs_shape[0], *obs_shape[1:])

        self.rollouts = RolloutStorage(
            self.num_steps_per_rollout,
            self.num_parallel,
            obs_shape,
            self.actor_critic.actor.action_dim,
            self.actor_critic.state_size,
        )

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

        self.action_steps = self.env.config['action_step']
        self.action_rgr_steps = [self.action_steps.index(s) + 1 for s in self.env.config.get('action_rgr_step', [])]
        self.action_mask = torch.zeros(self.mini_batch_size, len(self.action_steps) + 1, self.env.frame_dim).to(self.device)

        if len(self.action_rgr_steps) > 0:
            self.action_mask[:, self.action_rgr_steps] = 1
        self.action_mask = self.action_mask.view(self.mini_batch_size, -1)
        self.actor_reg_weight = config.get('actor_reg_weight', 1)
        self.actor_bound_weight = config.get('actor_bound_weight', 0.0)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)
        if not self.env.is_rendered:
            self.logger = logging_util.wandbLogger(
                run_name=env.int_output_dir,
                proj_name="HCONTROL_{}_{}_{}_{}_{}".format(
                    self.NAME, env.NAME, env.model.NAME, env.dataset.NAME, self.NAME
                )
            )

        self.block_size = config['block_size']
        self.use_physics_constraints = config.get('use_physics_constraints', True)
        self.use_bmdm_integration = config.get('use_bmdm_integration', True) and bmdm_model is not None

        if self.use_physics_constraints:
            self.physics_reward = PhysicsConstraintReward(config, env.dataset, device)

        if self.use_bmdm_integration:
            self.bmdm_explorer = BMDMExplorer(config, bmdm_model, env.dataset, device)
            self.bmdm_guide = BMDMGuide(config, bmdm_model, env.dataset, device)

        self.prev_pose = None
        self.physics_weight = config.get('physics_weight', 1.0)
        self.bmdm_guide_weight = config.get('bmdm_guide_weight', 0.5)

    def compute_action_bound_loss(self, norm_a, bound_min=-1, bound_max=1):
        violation_min = torch.clamp_max(norm_a.mean() - bound_min, 0.0)
        violation_max = torch.clamp_min(norm_a.mean() - bound_max, 0)
        bound_violation_loss = torch.sum(torch.square(violation_min), dim=-1) \
                    + torch.sum(torch.square(violation_max), dim=-1)
        return bound_violation_loss.mean()

    def compute_action_reg_weight(self, norm_a, mask):
        norm_a = norm_a * mask
        action_reg_loss = torch.sum(torch.square(norm_a), dim=-1)
        return action_reg_loss.mean()

    def test_controller(self):
        self.num_parallel = self.env.num_parallel_test
        obs = self.env.reset()
        ep_reward = 0

        self.env.reset_initial_frames()

        if self.use_physics_constraints:
            self.physics_reward.reset_history(self.num_parallel)

        with EpisodeRunner(self.env) as runner:
            while not runner.done:
                with torch.no_grad():
                    action = self.actor_critic.actor(obs)

                obs, reward, done, info = self.env.step(action)
                ep_reward += reward

                if done.any():
                    print("--- Episode reward: %2.4f" % float(ep_reward[done].mean()))
                    ep_reward *= (~done).float()
                    reset_indices = self.env.parallel_ind_buf.masked_select(done.squeeze())
                    obs = self.env.reset_index(reset_indices)

                if info.get("reset"):
                    print("--- Episode reward: %2.4f" % float(ep_reward.mean()))
                    ep_reward = 0
                    obs = self.env.reset()

    def train_controller(self, out_model_file, int_output_dir):
        obs = self.env.reset()
        self.rollouts.observations[0].copy_(obs)
        self.rollouts.to(self.device)
        num_samples = 0

        if self.use_physics_constraints:
            self.physics_reward.reset_history(self.num_parallel)

        self.prev_pose = None

        for update in range(self.num_updates):
            ep_info = {"reward": [], "physics_reward": [], "bmdm_guide": []}
            ep_reward = 0

            if self.lr_decay_type == "linear":
                update_linear_schedule(
                    self.optimizer, update, self.num_updates, self.lr, self.final_lr
                )
            elif self.lr_decay_type == "exponential":
                update_exponential_schedule(
                    self.optimizer, update, 0.99, self.lr, self.final_lr
                )

            for step in range(self.num_steps_per_rollout):
                with torch.no_grad():
                    value, action, action_log_prob = self.actor_critic.act(
                        self.rollouts.observations[step]
                    )

                obs, reward, done, info = self.env.step(action)

                current_pose = self.env.history[:, 0]

                additional_rewards = {}

                if self.use_physics_constraints:
                    physics_reward, physics_components = self.physics_reward(current_pose, self.prev_pose)
                    reward += self.physics_weight * physics_reward
                    additional_rewards['physics'] = physics_components

                if self.use_bmdm_integration:
                    cond_frame = self.env.get_cond_frame()
                    guide_reward, similarity = self.bmdm_guide(action, cond_frame)
                    reward += self.bmdm_guide_weight * guide_reward
                    additional_rewards['bmdm_guide'] = similarity.mean().item()

                ep_reward += reward

                end_of_rollout = info.get("reset")

                masks = (~done).float()
                bad_masks = (~(done * end_of_rollout)).float()

                if done.any():
                    ep_info["reward"].append(ep_reward[done].clone())
                    if 'physics' in additional_rewards:
                        ep_info["physics_reward"].append(additional_rewards['physics']['total_physics'][done].clone())
                    if 'bmdm_guide' in additional_rewards:
                        ep_info["bmdm_guide"].append(torch.tensor([additional_rewards['bmdm_guide']] * done.sum(), device=self.device))
                    ep_reward *= (~done).float()
                    reset_indices = self.env.parallel_ind_buf.masked_select(done.squeeze())
                    obs = self.env.reset_index(reset_indices)

                    if self.use_physics_constraints:
                        self.physics_reward.reset_history(len(reset_indices))

                if torch.all(end_of_rollout):
                    obs = self.env.reset()

                self.rollouts.insert(
                    obs, action, action_log_prob, value, reward, masks, bad_masks
                )

                self.prev_pose = current_pose

            num_samples += (obs.shape[0] * self.num_steps_per_rollout)
            with torch.no_grad():
                next_value = self.actor_critic.get_value(self.rollouts.observations[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda)

            value_loss, action_loss, dist_entropy, regr = self.update(self.rollouts)

            self.rollouts.after_update()

            save_util.save_weight(copy.deepcopy(self.actor_critic), out_model_file)
            if update % self.save_interval == 0:
                save_util.save_weight(copy.deepcopy(self.actor_critic), int_output_dir + '/_ep{}.pth'.format(update))

            stats = {
                "update": update,
                "value_loss": value_loss,
                "action_loss": action_loss,
                "dist_entropy": dist_entropy,
                "regr": regr,
            }

            if len(ep_info["reward"]) > 0:
                ep_info["reward"] = torch.cat(ep_info["reward"])
                stats.update({
                    "reward_mean": torch.mean(ep_info['reward']),
                    "reward_max": torch.max(ep_info['reward']),
                    "reward_min": torch.min(ep_info['reward']),
                })

            if len(ep_info["physics_reward"]) > 0:
                stats["physics_reward_mean"] = torch.mean(torch.cat(ep_info["physics_reward"]))

            if len(ep_info["bmdm_guide"]) > 0:
                stats["bmdm_similarity"] = torch.mean(torch.cat(ep_info["bmdm_guide"]))

            self.logger.log_epoch(stats, step=int(num_samples))
            self.logger.print_log(stats)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
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

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    observations_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )

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
