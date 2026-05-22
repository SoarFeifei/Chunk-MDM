import os.path as osp
import numpy as np
import gymnasium as gym

import torch
from render.realtime.mocap_renderer import PBLMocapViewer

coord_table = {'x':0, 'y':2, 'z':1}
def get_xyz_index(coord_order):
    return [coord_table[i] for i in coord_order]    

class EnvBase(gym.Env):
    def __init__(self, config, model, dataset, device):
        self.device = device
        self.is_rendered = config.get('is_rendered',True)
        self.num_parallel = config.get('num_parallel',1)
        self.num_parallel_test = config.get('num_parallel_test',1)

        self.frame_skip = config.get('frame_skip',1)
        self.max_timestep = config.get('max_timestep_test',2000) if self.is_rendered else config.get('max_timestep',1000//self.frame_skip)
        self.camera_tracking = config.get('camera_tracking',True)
        
        self.int_output_dir = config['int_output_dir']

        self.block_size = config['block_size']

        self.model = model
        self.dataset = dataset       

        self.frame_dim = dataset.frame_dim
        # print("env self.frame_dim", self.frame_dim)     # 1335
        self.data_fps = dataset.fps
        
        self.sk_dict = dataset.skel_info
        
        self.links = dataset.links
        self.name_joint = dataset.joint_names
        self.offset_joint = dataset.joint_offset
        self.num_joint = len(self.name_joint)

        self.root_idx = dataset.root_idx
        self.foot_idx = dataset.foot_idx
                
        self.action_scale = config.get('action_scale',1.0)
        self.test_action_scale = config.get('test_action_scale',self.action_scale)

        self.model_type = config['model_type']
        if config['model_type'] == 'amdm':
            
            self.action_step = config['action_step']
            self.use_action_mask = config.get('use_action_mask',False)
            
            if len(config['action_step']) == 0:
                self.action_step = list(range(model.T))
            
            self.action_mode = config['action_mode']
            self.random_scale = config['random_scale']
            self.test_random_scale = config['test_random_scale']

            self.clip_scale = config.get('clip_scale',2.5)
            if self.action_mode == 'loco':
                self.action_dim_per_step = 8
                
            elif self.action_mode == 'full':
                self.action_dim_per_step = self.frame_dim 
            
            self.action_dim = self.frame_dim + self.action_dim_per_step * len(self.action_step) ### frame_dim * (1 + 13)
            # self.action_dim = self.frame_dim    ###
            # self.action_dim = self.block_size * 3

            self.extra_info = {'action_step':self.action_step,
                               'action_mode':self.action_mode, 
                               'is_train': not self.is_rendered, 
                               'action_scale': self.action_scale,
                               'test_action_scale': self.test_action_scale,
                               'rand_scale':self.random_scale, 
                               'test_rand_scale':self.test_random_scale,
                               'clip_scale':self.clip_scale}
            


        elif config['model_type'] == 'humor':   
            self.action_dim = model.action_dim #if hasattr(self.model,'action_dim') else 64
            self.extra_info = None
        
        elif config['model_type'] == 'mvae':   
            self.action_dim = model.action_dim #if hasattr(self.model,'action_dim') else 64
            self.extra_info = None

        else:
            self.action_dim = dataset.frame_dim
            self.extra_info = None

        if self.is_rendered:
            self.record_num_frames = np.zeros((self.num_parallel_test,))
            self.record_motion_seq = np.zeros((self.num_parallel_test, self.max_timestep, self.dataset.frame_dim))
            self.record_timestep = 0

        # history size is used to calculate floating as well
        self.history_size = 5
        self.num_condition_frames = 1
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_dim)
        ).to(self.device)

        self.init_frame = torch.zeros((self.num_parallel, self.frame_dim)).to(self.device)
        self.timestep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.substep = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_facing = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_xz = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.root_y = torch.zeros((self.num_parallel, )).to(self.device)

        self.reward = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.potential = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)
        self.early_stop = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        # used for reward-based early termination
        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )
        
        if self.is_rendered:
            self.viewer = PBLMocapViewer(
                    self,
                    num_characters=self.num_parallel,
                    target_fps=self.data_fps,
                    camera_tracking=self.camera_tracking,
                )
            
        high = np.inf * np.ones([self.action_dim])  ### per frame int(self.action_dim / self.block_size)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        ###
        self.root_facing_seq = torch.zeros((self.block_size, self.num_parallel, 1)).to(self.device)
        self.root_xz_seq = torch.zeros((self.block_size, self.num_parallel, 2)).to(self.device)


    def save_motion(self):
        seqs = self.dataset.denorm_data(self.record_motion_seq)#.detach().cpu().numpy()
        for i in range(seqs.shape[0]):
            seq = seqs[i]
            xzs = self.dataset.x_to_trajs(seq)
            self.dataset.save_bvh(osp.join(self.int_output_dir,'out{}'.format(i)),seq)
            np.save(osp.join(self.int_output_dir,'traj{}'.format(i)),xzs)

        np.savez(osp.join(self.int_output_dir,'out.npz'), action=None, init_frame = self.init_frame.cpu().numpy(), nframe=self.record_timestep)



    def integrate_root_translation(self, pose):
        block_size = self.block_size

        ### per frame
        # pose_first = pose[:, :int(pose.shape[-1] / block_size)]
        # # print("pose_first", pose_first.shape)
        # pose_denorm = self.dataset.denorm_data(pose_first, device=pose.device)
        # dr = self.dataset.get_heading_dr(pose_denorm)[..., None]
        # root_xz_vel = self.dataset.get_root_linear_planar_vel(pose_denorm)
        #
        # root_rotmat_up = self.get_rotation_matrix(self.root_facing)
        # # print("root_xz_vel", root_xz_vel.shape)     # root_xz_vel torch.Size([512, 2])
        # # print("root_rotmat_up", root_rotmat_up.shape)   # root_rotmat_up torch.Size([512, 2, 2])
        # displacement = (root_rotmat_up * root_xz_vel.unsqueeze(1)).sum(dim=2)
        # # print("displacement", displacement.shape)   # torch.Size([512, 2])
        # self.root_facing.add_(dr).remainder_(2 * np.pi)
        # self.root_xz.add_(displacement)
        original_root_facing = self.root_facing
        original_root_xz = self.root_xz
        for i in range(block_size):
            single_dim = int(self.frame_dim / self.block_size)
            pose_t = pose[:, i * single_dim:(i + 1) * single_dim]
            pose_denorm = self.dataset.denorm_data(pose_t, device=pose.device)
            dr = self.dataset.get_heading_dr(pose_denorm)[..., None]
            root_xz_vel = self.dataset.get_root_linear_planar_vel(pose_denorm)

            # 核心修复 1：【先】更新全局朝向
            # 对应 dataset 中的 yaws[i] = yaws[i-1] + dr[i]
            # ---------------------------------------------------------
            self.root_facing.add_(dr).remainder_(2 * np.pi)

            root_rotmat_up = self.get_rotation_matrix(self.root_facing)
            # print("root_xz_vel", root_xz_vel.shape)     # root_xz_vel torch.Size([512, 2])
            # print("root_rotmat_up", root_rotmat_up.shape)   # root_rotmat_up torch.Size([512, 2, 2])
            displacement = (root_rotmat_up * root_xz_vel.unsqueeze(1)).sum(dim=2)
            # print("displacement", displacement.shape)   # torch.Size([512, 2])

            # self.root_facing.add_(dr).remainder_(2 * np.pi)   ###

            # print("self.root_facing", self.root_facing[0])
            self.root_xz.add_(displacement)
            # print("self.root_xz", self.root_xz[0])
            self.root_facing_seq[i] = self.root_facing
            self.root_xz_seq[i] = self.root_xz


        # print("poses 22", poses.shape)    # torch.Size([512, 1335])
        ###
        # # print("pose", pose.shape)   # pose torch.Size([512, 1335])

        ### per frame
        # per_frame_dim = int(pose.shape[-1] / self.block_size)
        # last_block = self.history[:, :self.num_condition_frames].view(pose.shape[0], pose.shape[-1])
        # pose = torch.cat([last_block[:, per_frame_dim:self.block_size * per_frame_dim], pose[:, 0:per_frame_dim]], dim=1)
        # print(self.history[:, :self.num_condition_frames].shape, last_block.shape, pose.shape)

        self.history = self.history.roll(1, dims=1)
        self.history[:, :self.num_condition_frames].copy_(pose.view(pose.shape[0], -1, pose.shape[-1]))     ###
        # print("self.history[:, :self.num_condition_frames]",self.history[:, :self.num_condition_frames].shape)  # torch.Size([512, 1, 1335])



    def get_rotation_matrix(self, yaw, dim=2):
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        if dim == 3:
            col1 = torch.cat((yaw.cos(), yaw.sin(), zeros), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos(), zeros), dim=-1)
            col3 = torch.cat((zeros, zeros, ones), dim=-1)
            matrix = torch.stack((col1, col2, col3), dim=-1)
        else:
            col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
            matrix = torch.stack((col1, col2), dim=-1)
        return matrix


    def get_cond_frame(self):
        condition = self.history[:, :self.num_condition_frames].view(-1, self.frame_dim)
        # print("history", self.history.shape)    # history torch.Size([512, 5, 1335])
        return condition

    def get_next_frame(self, action):       ### get_next_block
        self.action = action
        condition = self.get_cond_frame()
        # print("condition", condition.shape)     ###condition torch.Size([512, 267])
        extra_info = self.extra_info
        # print("action", action.shape)
        # print("extra_info", extra_info)

        with torch.no_grad():
            output = self.model.rl_step(condition, action, extra_info)
            
        #if self.is_rendered:   
        #    self.record_motion_seq[:,self.record_timestep,:]= output.cpu().detach().numpy()
        #    self.record_timestep += 1
        #    if self.record_timestep % 90 == 0 and self.record_timestep != 0:
        #        self.save_motion()

        # print("output",output.shape)    ### output torch.Size([512, 267])
        return output

    def reset(self):
    
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.reward.fill_(0)
        self.timestep.fill_(0)
        self.substep.fill_(0)
        self.done.fill_(False)
        # value bigger than contact_threshold
        #self.foot_pos_history.fill_(1)

        self.reset_target()
        self.reset_initial_frames()
        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)


    def reset_index(self, indices=None):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            self.done.fill_(False)
            # value bigger than contact_threshold
            #self.foot_pos_history.fill_(1)

            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.reset_target(indices)
            self.reset_initial_frames(indices)
            # value bigger than contact_threshold
            #self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)


    def reset_initial_frames(self, index=None, rl_list=None):
        # Make sure condition_range doesn't blow up
        num_init = self.num_parallel if index is None else len(index)
        #$$
        start_index = torch.randint(0, len(self.dataset.valid_idx), (num_init,1))

        start_index = self.dataset.valid_idx[start_index] if rl_list is None else rl_list
        # print("start_index", start_index)
        # print("self.dataset.motion_flattened", self.dataset.motion_flattened.shape)
        data = torch.tensor(self.dataset.motion_flattened[start_index], device = self.device, dtype=torch.float32).clone()
    
        # if self.is_rendered:
            # print('resetting, starting frame index:',start_index)

        if not index:
            #self.init_frame[:] = data.squeeze()
            self.history[:, :self.num_condition_frames].copy_(data)
        else:
            #self.init_frame[index] = data.squeeze()
            self.history[index, :self.num_condition_frames].copy_(data)


    def calc_foot_slide(self):
        return 0
        '''
        foot_z = self.foot_pos_history[:, :, [2, 5]]
        # in_contact = foot_z < self.contact_threshold
        # contact_coef = in_contact.all(dim=1).float()
        # foot_xy = self.foot_pos_history[:, :, [[0, 1], [3, 4]]]
        # displacement = (
        #     (foot_xy.unsqueeze(1) - foot_xy.unsqueeze(2))
        #     .norm(dim=-1)
        #     .max(dim=1)[0]
        #     .max(dim=1)[0]
        # )
        # foot_slide = contact_coef * displacement

        displacement = self.foot_pos_history[:, 0] - self.foot_pos_history[:, 1]
        displacement = displacement[:, [[0, 1], [3, 4]]].norm(dim=-1)

        foot_slide = displacement.mul(
            2 - 2 ** (foot_z.max(dim=1)[0] / self.contact_threshold).clamp_(0, 1)
        )
        return foot_slide
        '''

    def calc_rigid_penalty(self):
        pass

    def calc_jittering(self):
        return 0

    def calc_energy_penalty(self, next_frame):
        vel_dim_lst = self.dataset.vel_dim_lst
        action_energy = (
            next_frame[:, [0, 1]].pow(2).sum(1)
            + next_frame[:, 2].pow(2)
            + next_frame[:,  vel_dim_lst[0]:  vel_dim_lst[1]].pow(2).mean(1)
        )
        return -0.8 * action_energy.unsqueeze(dim=1)

    def calc_action_penalty_reward(self):
        prob_energy = self.action.abs().mean(-1, keepdim=True)
        return -0.01 * prob_energy


    def step(self, action):

        # fake_action = torch.tensor([0.0, -1.0, 0.0] * self.block_size, device=self.device)
        # fake_action = fake_action.unsqueeze(0).expand(action.shape[0], -1)
        # print("fake_action", fake_action.shape)
        # print("self.root_xz", self.root_xz)

        next_frame = self.get_next_frame(action)    # next_block
        # print("next_frame",next_frame.shape)    ### next_frame torch.Size([512, 267 * n])

        # 检查是否包含 NaN, Inf，或者数值绝对值超过了物理极限 (比如 z-score > 20.0)
        is_exploded = torch.isnan(next_frame).any(dim=-1, keepdim=True) | \
                      torch.isinf(next_frame).any(dim=-1, keepdim=True) | \
                      (next_frame.abs().max(dim=-1, keepdim=True)[0] > 20.0)
        # 如果当前环境爆炸了，用一个全零的绝对安全帧强行覆盖，防止 NaN 传染给神经网络
        safe_frame = torch.zeros_like(next_frame)
        next_frame = torch.where(is_exploded, safe_frame, next_frame)
        # print("next_frame", next_frame.abs().max().item())

        obs, reward, done, info = self.calc_env_state(next_frame)
        # print("done", done.shape, next_frame.abs().max().item(), is_exploded.shape)
        # print("obs", obs.shape)     # torch.Size([512, 803]) 267 * 3 + 2
        # print("self.max_timestep",self.max_timestep)    ### 300

        # 【关键】：让爆炸的环境立刻强制结束回合 (Done=True)，防止它继续积累误差
        done = done.squeeze(-1)
        is_exploded = is_exploded.squeeze(-1)
        done = done | is_exploded
        done = done.unsqueeze(-1)

        return (obs, reward, done, info)
        
        
    def calc_env_state(self, next_frame):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.is_rendered:
            self.viewer.close()

    def render(self, mode="human"):
        # frame = self.dataset.denorm_data(self.history[:, 0], device=self.device).detach().cpu().numpy()
        # if self.is_rendered:
        #     self.viewer.render(
        #         torch.tensor(self.dataset.x_to_jnts(frame, mode='angle'),device=self.device,dtype=self.root_facing.dtype),  # 0 is the newest
        #         self.root_facing,
        #         self.root_xz,
        #         0.0,  # No time in this env
        #         self.action,
        #     )

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
        #         self.action,
        #     )
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
                    self.action,
                )