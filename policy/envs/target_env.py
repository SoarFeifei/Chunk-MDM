
import policy.envs.base_env as base_env
from render.realtime.mocap_renderer import PBLMocapViewer
import torch
import numpy as np
import tkinter as tk
import gymnasium as gym

from multiprocessing import Process
from filelock import FileLock
#user_input_lockfile = "miscs/interact_temp/user_text"
import os.path as osp

class TargetEnv(base_env.EnvBase):
    NAME = 'Target'
    def __init__(self, config, model, dataset, device):      
        super().__init__(config, model, dataset, device)
        self.device = device
        self.config = config
        self.model = model
        self.dataset = dataset

        self.links = self.dataset.links
        self.valid_idx = self.dataset.valid_idx
        
        self.index_of_target = 0
        self.arena_length = (-7.0, 7.0)
        self.arena_width = (-7.0, 7.0)

        self.num_future_predictions = 1
        self.num_condition_frames = 1
        
        target_dim = 2
        
        self.target = torch.zeros((self.num_parallel, target_dim)).to(self.device)      # (512,2)
        self.observation_dim = (self.frame_dim * self.num_condition_frames) #+ target_dim    ### per frame
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        high = np.inf * np.ones([self.action_dim])   ### per frame int(self.action_dim / self.block_size)
        # print("high", high.shape)
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.target_arr = torch.zeros((self.num_parallel, self.max_timestep, 3))

        self.next = False    ###

    def calc_potential(self):
        target_delta, target_angle = self.get_target_delta_and_angle()
        self.linear_potential = -target_delta.norm(dim=1).unsqueeze(1)
        self.angular_potential = target_angle.cos()

    def get_target_delta_and_angle(self):
        # print("111")
        target_delta = self.target - self.root_xz
        # print("self.target", self.target[0])
        # print("self.root_xz", self.root_xz[0])
        target_angle = (
            torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1)
            + self.root_facing
        )

        return target_delta, target_angle

    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        #Should be negative because going from global to local

        mat = self.get_rotation_matrix(-self.root_facing)
        # print("target_delta", target_delta.shape)

        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)
        # print("mat, delta", mat.shape, delta.shape)   # mat, delta torch.Size([512, 2, 2]) torch.Size([512, 2])
        condition = self.get_cond_frame()
        # if self.next:
        #     condition = self.next_frame    ###
        # print("condition delta",condition.shape,delta.shape)    ### condition delta torch.Size([512, 267]) torch.Size([512, 2])
        return condition#, delta
    
    def reset(self, indices=None):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            self.done.fill_(False)

            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.timestep.fill_(0)
            self.substep.fill_(0)
            
            self.reset_target(indices)

        obs_components = self.get_observation_components()
        # return torch.cat(obs_components, dim=1)   ###
        return obs_components
    
    def reset_index(self, indices=None):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep = 0
            self.substep = 0
            self.done.fill_(False)
            
            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.reset_target(indices)

            # value bigger than contact_threshold
            #self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)
        obs_components = self.get_observation_components()
        # return torch.cat(obs_components, dim=1)   ###
        return obs_components

    def output_motion(self):
        #flag_pos_hist = np.array(self.flag_pos.detach().cpu())
        f = open('./flag.txt','w')
        for st,ed in self.flag_sted:
            f.write("{},{}\n".format(st,ed))
        f.close()
        #np.savez(file='../../bvh_demo/out_info.npz',flag_pos=flag_pos_hist,sted=self.flag_sted)
        return super().output_motion()

    def calc_action_penalty_reward(self):
        prob_energy = self.action[..., self.action_dim_per_step:].abs().mean(-1, keepdim=True)
        return -0.02 * prob_energy
    
    def reset_initial_frames(self, frame_index=None):
        # Make sure condition_range doesn't blow up
        num_frame_used = len(self.valid_idx)
        num_init = self.num_parallel if frame_index is None else len(frame_index)

        #ensor([[537085]]) ==================
        #tensor([[2122372]]) ==================
        
        start_index = torch.randint(0,num_frame_used-1,(num_init,1))#2122372 #torch.randint(0,num_frame_used-1,(num_init,1)) 
        start_index = self.valid_idx[start_index]
        # print("start_index", start_index[[75, 127, 210, 237, 241],:])
        # large_index = [[92717],[3873],[44707],[27030],[27394]]
        # start_index[-5:,] = large_index

        data = torch.tensor(self.dataset.motion_flattened[start_index])
        # print('starting index:', start_index.shape)
        print("self.dataset.motion_flattened", self.dataset.motion_flattened.shape)
        # print('starting index:', start_index)
        if self.is_rendered:
            print('starting index:',start_index)
        if frame_index is None:
            self.init_frame = data.clone()
            self.history[:, :self.num_condition_frames] = data.clone()
        else:
            self.init_frame[frame_index] = data.clone()
            self.history[frame_index, :self.num_condition_frames] = data.clone()

    
    def reset_target(self, indices=None, location=None):
        if location is None:
            if indices is None:
                self.target[:, 0].uniform_(*self.arena_length) 
                self.target[:, 1].uniform_(*self.arena_width)
                self.target[:, 0] += + self.root_xz[:,0]
                self.target[:, 1] += + self.root_xz[:,1]
            else:
                # if indices is a pytorch tensor, this returns a new storage
                new_lengths = self.target[indices, 0].uniform_(*self.arena_length) + self.root_xz[indices,0]  ### self.root_xz[:,0]
                self.target[:, 0].index_copy_(dim=0, index=indices, source=new_lengths)
                new_widths = self.target[indices, 1].uniform_(*self.arena_width) + self.root_xz[indices,1]  ### self.root_xz[:,0]
                self.target[:, 1].index_copy_(dim=0, index=indices, source=new_widths)

            
        else:
            # Reaches this branch only with mouse click in render mode
            self.target[:, 0] = location[0]
            self.target[:, 1] = location[1]
        
        """ 
        targets_lst =torch.tensor([[-5.0,10.0],
                      [ 5.0, 6.0],
                      [ 2.0, 2.0],
                      [1,1],[5,5],[1,1],[6,6],[0,0],[0,5], #], device=self.device)
                      [ -12.2380, -5.6012],
                      [-5.2380, -10],
                      [3,-11],
                      [5,-10],
                      [10,5],
                      [0,0]
                      ]).to(self.device) 
        self.target = targets_lst[None,self.index_of_target]               
        """
        

        if self.is_rendered:
            self.target_arr[...,self.index_of_target,:2] = self.target[:, :2]#.detach().cpu().numpy()
            # print(self.target_arr[...,self.index_of_target,2].shape, self.timestep.shape)
            self.target_arr[...,self.index_of_target,2] = self.timestep.squeeze(-1)
            self.timestep.unsqueeze(-1)
            self.index_of_target += 1
            
            np.save(osp.join(self.int_output_dir,'out_target'), self.target_arr)
            self.viewer.update_target_markers(self.target)

        # Should do this every time target is reset
        self.calc_potential()

    def calc_progress_reward(self):
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential

        self.calc_potential()
        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential
        progress = linear_progress
        
        return progress

    def calc_env_state(self, next_frame):
        self.next = True
        self.next_frame = next_frame
        # print("next_frame", next_frame.shape)
        is_external_step = self.substep == 0

        if torch.all(self.substep == self.frame_skip - 1):
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)
        progress = self.calc_progress_reward()    ###
        # progress = torch.clamp(progress, min=0.0)    ###
        # print("progress", progress.shape, progress.min().item(), progress.max().item())
       
        target_dist = -self.linear_potential
        target_is_close = (target_dist < 0.4)
        # print("target_is_close",target_is_close.shape)    #torch.Size([512],1)
        # print("self.linear_potential", self.linear_potential.shape)
        # print("target_dist", target_dist.max().item(), target_dist.min().item())
        # # print("target_is_close", target_is_close)
        # print("progress", progress[0], self.target[0], self.root_xz[0])
        dist_reward = 2 * torch.exp(0.5 * self.linear_potential) + progress * 2

        self.reward.copy_(dist_reward)

        # large_reward_mask = self.reward < -10000
        # large_indices = large_reward_mask.nonzero()[:, 0]
        # if(large_indices.shape[0]>0):
        #     print("root_xz", large_indices, self.root_xz[large_indices, :])

        # self.reward[target_is_close].copy_(5)
        # if target_is_close.any():
        #     self.reward[target_is_close] = 5.0
        self.reward.masked_fill_(target_is_close, 15)
        # print("true_count", self.reward[target_is_close], self.reward.max().item(), self.reward.min().item())

        if target_is_close.any() and self.is_rendered:
            reset_indices = self.parallel_ind_buf.masked_select(
                target_is_close.squeeze(-1)
            )
            print("root_xz", self.root_xz[reset_indices], self.target[reset_indices])
            self.reset_target(indices=reset_indices)
            #self.steps_parallel[reset_indices.cpu().detach()] *= 0

        obs_components = self.get_observation_components()
        # print("self.timestep >= self.max_timestep",self.timestep >= self.max_timestep)
        # self.done.fill_(self.timestep >= self.max_timestep) ###
        done_mask = (self.timestep >= self.max_timestep)
        # print("done_mask", done_mask.shape)
        self.done.copy_(done_mask)
        # self.done.copy_(self.timestep >= self.max_timestep) ###

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()
        # print("obs_components", obs_components[0].shape)  # obs_components[0] = obs ,obs_components = target



        return (
            # torch.cat(obs_components, dim=1),   ###
            obs_components,
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep,
            
            },
        )

    def dump_additional_render_data(self):
        return {"extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]}}

        # if self.is_rendered and self.timestep % 10 == 0:
        #     self.viewer.duplicate_character()

