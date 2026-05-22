
import policy.envs.base_env as base_env
from render.realtime.mocap_renderer import PBLMocapViewer
import torch
import numpy as np
import gymnasium as gym

class RandomPlayEnv(base_env.EnvBase):
    NAME = "RandomPlay"
    def __init__(self, config, model, dataset, device):
        self.device = device
        self.config = config
        self.model = model
        self.dataset = dataset

        self.interative_text = False
        self.cur_extra_info = None
        self.updated_text = False

        self.links = self.dataset.links
        self.valid_idx = self.dataset.valid_idx

        self.frame_dim = self.dataset.frame_dim
        self.action_dim = self.dataset.frame_dim
        self.valid_range = self.dataset.valid_range
        self.sk_dict = dataset.skel_info
        self.data_fps = self.dataset.fps

        self.is_rendered = True
        self.num_parallel = config.get('num_parallel', 1)
        self.frame_skip = config.get('frame_skip',1)
        self.max_timestep = config.get('max_timestep',10000)
        self.camera_tracking = config.get('camera_tracking',True)
        self.int_output_dir = config['int_output_dir']

        self.block_size = config['block_size']
        self.overlap = config['overlap']

        self.num_condition_frames = 1

        self.base_action = torch.zeros((self.num_parallel, 1, self.action_dim)).to(
            self.device
        )
        self.timestep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.substep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.reward = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_facing = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_xz = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.root_y = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        self.history_size = 5
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_dim)
        ).to(self.device)

        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )

        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.viewer = PBLMocapViewer(
            self,
            num_characters=self.num_parallel,
            target_fps=self.data_fps,
            camera_tracking=self.camera_tracking,
        )

        if self.is_rendered:
            self.record_num_frames = np.zeros((self.num_parallel,))
            self.record_motion_seq = np.zeros((self.num_parallel, self.max_timestep, self.dataset.frame_dim))

        ###
        self.root_facing_seq = torch.zeros((self.block_size, self.num_parallel, 1)).to(self.device)
        self.root_xz_seq = torch.zeros((self.block_size, self.num_parallel, 2)).to(self.device)

    def get_cond_frame(self):
        condition = self.history[:, : self.num_condition_frames]
        return condition.view(condition.shape[0],-1)
    

    def get_next_frame(self, action=None, conds=None):  ### conditions
        condition = self.get_cond_frame()
       
        with torch.no_grad():
            self.cur_extra_info = conds
            output = self.model.eval_step(condition, self.cur_extra_info)
            #output = self.dataset.unify_rpr_within_frame(condition, output)
        
        return output
    
    def reset(self, rl_list=None):
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.reward.fill_(0)
        self.timestep.fill_(0)
        self.substep.fill_(0)
        self.done.fill_(False)
        self.reset_initial_frames(rl_list=rl_list)


    def reset_index(self, indices):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            self.done.fill_(False)
            self.reset_initial_frames()

        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            self.reset_initial_frames(indices)

        return 

    def target_markers(self, targets):  ### conditions
        targets = targets.reshape(1, 2)
        self.viewer.update_target_markers(targets)
    def calc_env_state(self, next_frame):
        
        self.reward.fill_(1) 
        
        self.timestep[self.substep == self.frame_skip - 1] += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)
 
        #foot_slide = self.calc_foot_slide()

        self.done[self.timestep >= self.max_timestep] = True

        self.render()
        return (
            None,
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def step_rl(self, generated_motion):
        """
            专门为 DDPO 采样设计的步进函数
            generated_motion: [batch_size, block_size * single_frame_dim]
            """
        # 1. 更新根节点位置和朝向 (这一步会内部更新 self.history)
        # pose 就是生成的动作块
        self.integrate_root_translation(generated_motion)

        # 2. 计算环境状态（步数增加、检查是否结束）
        # 注意：calc_env_state 内部会处理 timestep 和 done
        obs, reward, done, info = self.calc_env_state(generated_motion)

        return obs, reward, done, info
    def render(self, mode="human"):
        # frame = self.dataset.denorm_data(self.history[:, 0], device=self.device).cpu().numpy()
        # self.viewer.render(
        #     torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'),device=self.device, dtype=self.history.dtype),  # 0 is the newest
        #     self.root_facing,
        #     self.root_xz,
        #     0.0,  # No time in this env
        #     0.0   #self.action,
        # )
        ###
        block_size = self.block_size
        history = self.history[:, 0]
        # frame = self.dataset.denorm_data(history[:, -267:], device=self.device).detach().cpu().numpy()
        # if self.is_rendered:
        #     self.viewer.render(
        #         torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'), device=self.device,
        #                      dtype=self.root_facing.dtype),  # 0 is the newest
        #         self.root_facing,
        #         self.root_xz,
        #         0.0,  # No time in this env
        #         0.0,
        #     )

        # block_size = self.block_size
        # history = self.history[:, 0]
        # for i in range(block_size):  # overlap
        #     frame = self.dataset.denorm_data(history[:, i * 267: (i + 1) * 267],
        #                                      device=self.device).detach().cpu().numpy()
        #     if self.is_rendered:
        #         self.viewer.render(
        #             torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'), device=self.device,
        #                          dtype=self.history.dtype),  # 0 is the newest
        #             self.root_facing,
        #             self.root_xz,
        #             0.0,  # No time in this env
        #             0.0
        #         )

        for i in range(block_size):
            single_dim = int(self.frame_dim / self.block_size)
            frame = self.dataset.denorm_data(history[:, i * single_dim: (i+1) * single_dim], device=self.device).detach().cpu().numpy()
            if self.is_rendered:
                self.viewer.render(
                    torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'), device=self.device,
                                 dtype=self.root_facing.dtype),  # 0 is the newest
                    self.root_facing_seq[i, :, :],
                    self.root_xz_seq[i, :, :],
                    0.0,  # No time in this env
                    0.0,
                )

    def dump_additional_render_data(self):
        pass
