import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from copy import deepcopy
import random

import model.model_base as model_base
import model.modules.EMA as EMA 
import model.modules.Embedding as Embedding
import model.modules.Activation as Activation

import dataset.util.geo as geo_util

import predictModel as predictModel
import time


class AMDM(model_base.BaseModel):
    NAME = 'AMDM'
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
       
        self.estimate_mode = config["diffusion"]["estimate_mode"]   
        self.loss_type = config["diffusion"]["loss_type"] 
        
        self.T = config["diffusion"]["T"] 
        self.sample_mode = config["diffusion"]["sample_mode"]  
        self.eval_T = config["diffusion"]["eval_T"] if self.sample_mode == 'ddim' else self.T #self.T

        self.frame_dim = dataset.frame_dim
        config['frame_dim'] = self.frame_dim
        self._build_model(config)

        ###
        self.dataset = dataset

        self.block_size = config["block"]["size"]
        self.overlap = config["block"]["overlap"]
        self.conds_flag = config["block"]["conds_flag"]

        self.use_dynamic_length = False  ### dynamic length

        self.use_ema = config["optimizer"].get("EMA", False)
        if self.use_ema:
            print("Using EMA")
            self.ema_step = 0
            self.ema_decay = config['optimizer']['EMA']['ema_decay']
            self.ema_start = config['optimizer']['EMA']['ema_start']
            self.ema_update_rate = config['optimizer']['EMA']['ema_update_rate']
            self.ema_diffusion = deepcopy(self.diffusion)
            self.ema = EMA.EMA(self.ema_decay)
        return

    def forward(self, input_lastx, input_noises, input_ts):
        x = input_noises[:, self.T]
        # print("x, x.shape, t ",x ,x.shape ,input_ts[:, 0])
        for t in range(self.T - 1, -1, -1):
            ts = input_ts[:, t]
            te = self.diffusion.time_mlp(ts)
            pred = self.diffusion.model(input_lastx, x, te)
            x = self.diffusion.remove_noise(x, pred, ts)
            if t > 0:
                x = self.diffusion.add_noise_w(x, ts, input_noises[:,t])
        return x

    def _build_model(self, config):
        self.diffusion = GaussianDiffusion(config)
        self.diffusion.to(self.device)
        return

    def eval_step(self, cur_x, extra_dict=None, align_rpr=False, record_process=False): 
        diffusion = self.ema_diffusion if self.use_ema else self.diffusion  
        with torch.no_grad():
            if self.sample_mode == 'ddpm':
                # start = time.time()
                next_x = diffusion.sample_ddpm(cur_x, extra_dict, record_process)
                # end = time.time()
                # print(f"Block-AMDM generate 1step in {end - start:.3f}s")

                # next_x = diffusion.sample_ddpm_best_of_n(cur_x, extra_dict)
                # print("BON")
                ###
                # if self.use_dynamic_length:
                #     num_trials = next_x.shape[0]
                #     max_len = self.block_size
                #     TsEncoder = predictModel.TemporalTransformerEncoder(input_dim=267, hidden_dim=512).to(self.device)
                #     PrefixPredictModel = predictModel.PrefixPredictor(hidden_dim=512, max_len=max_len).to(self.device)
                #     trained_model_path = "output/base/amdm_lafan1/"
                #     enc_state_dict = torch.load(trained_model_path + "encoder_param.pth")
                #     TsEncoder.load_state_dict(enc_state_dict)
                #     pre_state_dict = torch.load(trained_model_path + "predicitor_param.pth")
                #     PrefixPredictModel.load_state_dict(pre_state_dict)
                #
                #     pred_index = torch.zeros(num_trials, dtype=torch.long, device=self.device)
                #     output_xs = torch.zeros((num_trials, num_steps, int(self.frame_dim / self.block_size)),
                #                             device=self.device)
                #     while (pred_index < num_steps).any():
                #         with torch.no_grad():
                #             start_x = start_x.reshape(num_trials, self.frame_dim)
                #             prev_x = start_x.reshape(num_trials, self.block_size, int(self.frame_dim / self.block_size))
                #
                #             start_x = diffusion.sample_ddpm(cur_x, extra_dict, record_process)
                #             start_x = start_x.reshape(num_trials, self.block_size, -1)
                #
                #             h_seq, h_global = TsEncoder(prev_x)  # 编码上一块
                #             prob_k = PrefixPredictModel(h_global)  # 预测有效帧数分布
                #             k_index = torch.arange(prob_k.shape[1], device=prob_k.device).float()
                #             expected_len = (prob_k * k_index).sum(dim=-1)  # [B]
                #             expected_len = torch.round(expected_len).int()
                #             mask = (expected_len == 0)
                #             expected_len[mask] = 1
                #             print("expected_len", expected_len)
                #             for i in range(num_trials):
                #                 end_idx = min((pred_index[i] + expected_len[i]).int(), num_steps)
                #                 output_xs[i, pred_index[i]: end_idx, :] = start_x[i, : int(end_idx - pred_index[i]), :]
                #                 pred_index[i] = end_idx
                #                 temp = start_x[i, :expected_len[i], :]
                #                 if self.block_size > expected_len[i]:
                #                     start_x[i, :self.block_size - expected_len[i], :] = prev_x[i,
                #                                                                         self.block_size - expected_len[
                #                                                                             i]:, :]
                #                     start_x[i, expected_len[i]:, :] = temp
                #                 else:
                #                     start_x[i, :, :] = temp
                #             # for i in range(max_len):
                #             #     output_xs[:, pred_index + i, ...] = start_x[:, i, ...]
                #             # if expected_len < self.block_size:
                #             #     temp = start_x[:, :expected_len, :]
                #             #     start_x[:, :self.block_size - expected_len, :] = prev_x[:, self.block_size - expected_len:, :]
                #             #     start_x[:, expected_len:, :] = temp
                #             # pred_index += expected_len
                #         print("pred_index", pred_index)
                #     next_x = start_x
            elif self.sample_mode == 'ddim':
                next_x = diffusion.sample_ddim(cur_x, self.eval_T, 0.0, extra_dict)
            else:
                assert(False), "Unsupported agent: {}".format(self.estimate_mode)

        if align_rpr:
            next_x = self.align_frame_with_angle(cur_x, next_x).type(cur_x.dtype)
        # print("cur_x, x.shape, next_x.shape",cur_x ,cur_x.shape ,next_x.shape)
        return next_x

    def rl_step(self, start_x, action_dict, extra_dict):
        diffusion = self.ema_diffusion if self.use_ema else self.diffusion 
        return diffusion.sample_rl_ddpm(start_x, action_dict, extra_dict)

    # smooth_mask
    def boundary_mask(self, x, fade_len=2, mode='linear'):
        """
        x: [B, T, D] 一个block
        fade_len: 边界帧的淡入淡出长度
        mode: 'linear' 或 'hann'
        """
        # x = x.reshape(-1, self.block_size, int(x.shape[1] / self.block_size))
        B, T, D = x.shape
        weights = torch.ones(T, device=x.device)

        if fade_len > 0:
            if mode == 'linear':
                fade = torch.linspace(0.6, 0.8, steps=fade_len, device=x.device)
            elif mode == 'hann':  # 更平滑的余弦窗
                fade = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, steps=fade_len, device=x.device))

            # 淡入（前fade_len帧）
            weights[:fade_len] = fade
            # 淡出（后fade_len帧）
            weights[-fade_len:] = torch.flip(fade, dims=[0])

        mask_dims = list(range(10, 267))  # 除去 root 位置速度
        x[:, :, mask_dims] = x[:, :, mask_dims] * weights.unsqueeze(1)
        # 应用mask到所有特征维度
        return x * weights.unsqueeze(-1)

    def eval_seq(self, start_x, extra_dict, num_steps, num_trials, align_rpr=False, record_process=False):
    ###推理阶段生成未来序列帧，生成 num_steps 个未来预测
        if len(start_x.shape)<=1:
            start_x = start_x[None, :]
        
        if start_x.shape[0] == 1:
            start_x = start_x.expand(num_trials, -1)    #(num_trials, frame_dim * blocksize)
        else:
            print('overwrite num of trial with actual batch size of start_x')
            num_trials = start_x.shape[0]
        
        print("eval_seq start_x",start_x.shape)

        if record_process:
            output_xs = torch.zeros((num_trials, num_steps, self.T, int(self.frame_dim/self.block_size))).to(self.device)    ###
        else:
            output_xs = torch.zeros((num_trials, num_steps, int(self.frame_dim/self.block_size))).to(self.device)    ###

        ### conditions
        if self.conds_flag:
            traj_pose, traj_trans = extra_dict["traj_pose"].to(self.device), extra_dict["traj_trans"].to(
                self.device)

        ### dynamic length
        if self.use_dynamic_length:
            max_len = self.block_size
            TsEncoder = predictModel.TemporalTransformerEncoder(input_dim=267, hidden_dim=512).to(self.device)
            PrefixPredictModel = predictModel.PrefixPredictor(hidden_dim=512, max_len=max_len).to(self.device)
            trained_model_path = "output/base/amdm_lafan1/"
            enc_state_dict = torch.load(trained_model_path + "encoder_param.pth")
            TsEncoder.load_state_dict(enc_state_dict)
            pre_state_dict = torch.load(trained_model_path + "predicitor_param.pth")
            PrefixPredictModel.load_state_dict(pre_state_dict)

            pred_index = torch.zeros(num_trials, dtype=torch.long, device=self.device)
            output_xs = torch.zeros((num_trials, num_steps, int(self.frame_dim / self.block_size)), device=self.device)
            while (pred_index < num_steps).any():
                with torch.no_grad():
                    start_x = start_x.reshape(num_trials, self.frame_dim)
                    prev_x = start_x.reshape(num_trials, self.block_size, int(self.frame_dim / self.block_size))

                    start_x = self.eval_step(start_x, extra_dict, align_rpr, record_process).detach()
                    start_x = start_x.reshape(num_trials, self.block_size, -1)

                    h_seq, h_global = TsEncoder(prev_x)  # 编码上一块
                    prob_k = PrefixPredictModel(h_global)  # 预测有效帧数分布
                    k_index = torch.arange(prob_k.shape[1], device=prob_k.device).float()
                    expected_len = (prob_k * k_index).sum(dim=-1)  # [B]
                    expected_len = torch.round(expected_len).int()
                    mask = (expected_len == 0)
                    expected_len[mask] = 1
                    print("expected_len", expected_len)
                    for i in range(num_trials):
                        end_idx = min((pred_index[i] + expected_len[i]).int(), num_steps)
                        output_xs[i, pred_index[i]: end_idx, :] = start_x[i, : int(end_idx - pred_index[i]), :]
                        pred_index[i] = end_idx
                        temp = start_x[i, :expected_len[i], :]
                        if self.block_size > expected_len[i]:
                            start_x[i, :self.block_size - expected_len[i], :] = prev_x[i, self.block_size - expected_len[i]:, :]
                            start_x[i, expected_len[i]:, :] = temp
                        else:
                            start_x[i, :, :] = temp
                    # for i in range(max_len):
                    #     output_xs[:, pred_index + i, ...] = start_x[:, i, ...]
                    # if expected_len < self.block_size:
                    #     temp = start_x[:, :expected_len, :]
                    #     start_x[:, :self.block_size - expected_len, :] = prev_x[:, self.block_size - expected_len:, :]
                    #     start_x[:, expected_len:, :] = temp
                    # pred_index += expected_len
                # print("pred_index", pred_index)
        else:   ###
            for j in range(int(num_steps/self.block_size) + 1):
                with torch.no_grad():
                    ###
                    start_x = start_x.reshape(num_trials, self.frame_dim)

                    ### conditions
                    cond = None
                    if self.conds_flag:
                        cond = {}
                        cond["traj_pose"] = traj_pose[j, :].reshape(1, -1)
                        cond["traj_trans"] = traj_trans[j, :].reshape(1, -1)
                        # print("traj_pose", cond["traj_pose"], cond["traj_trans"])

                        # if num_trials > 1:
                        #     cond["traj_pose"] = cond["traj_pose"].repeat(num_trials, 1)
                        #     cond["traj_trans"] = cond["traj_trans"].repeat(num_trials, 1)
                        #     # cond["traj_pose"][:, :] -= 0.0001
                        #     if j == 4 or j == 8:
                        #         cond["traj_pose"][2, :4] += 0.004
                        #         cond["traj_pose"][4, :4] -= 0.004
                        #
                        #     if j == 6 or j == 9:
                        #         cond["traj_pose"][2, :4] -= 0.004
                        #         cond["traj_pose"][4, :4] += 0.004
                        #         cond["traj_pose"][3, 2:4] += 0.004
                        #         cond["traj_pose"][5, 2:4] -= 0.003
                        #     if j == 20 or j == 28:
                        #         cond["traj_pose"][6, 2:4] += 0.005
                        #         cond["traj_pose"][7, 2:4] -= 0.005
                    start_x = self.eval_step(start_x, cond, align_rpr, record_process).detach()
                    ###
                    start_x = start_x.reshape(num_trials, self.block_size, int(self.frame_dim/self.block_size))

                # smooth_mask
                # start_x = self.boundary_mask(start_x, 2)

                # overlap
                for i in range(self.block_size):    # overlap
                    if j * self.block_size + i < num_steps:
                        # print(j * (self.block_size - self.overlap) + i)
                        output_xs[:, j * self.block_size + i, ...] = start_x[:, i, ...]
                    # else:
                    #     print(j * (self.block_size - self.overlap) + i)

                # output_xs[:,j,...] = start_x

                ###
                # nan_count = torch.isnan(start_x).sum().item()
                # total = start_x.numel()
                # print(f"start_x has {nan_count} NaNs out of {total} elements")
                ###输出 start_x 会作为下一轮的输入（自回归）
                if record_process:
                    start_x = start_x[..., -1, :]
        ###
        # nan_count = torch.isnan(output_xs).sum().item()
        # total = output_xs.numel()
        # print(f"output_xs has {nan_count} NaNs out of {total} elements")

        return output_xs

    def eval_step_interactive(self, cur_x, edited_mask, edit_data, extra_dict): 
        diffusion = self.ema_diffusion if self.use_ema else self.diffusion

        if self.sample_mode == 'ddpm':
            return diffusion.sample_ddpm_interactive(cur_x, edited_mask, edit_data, extra_dict)
        #elif self.sample_mode == 'ddim':
        #    return self.model.sample_ddim_interactive(cur_x, self.eval_T, edited_data, edited_mask, extra_dict)
        else:
            assert(False), "Unsupported agent: {}".format(self.estimate_mode)                

    def eval_seq_interactive(self, start_x, extra_dict, edit_data, edited_mask, num_steps, num_trials):
        output_xs = torch.zeros((num_trials, num_steps, self.frame_dim)).to(self.device)
        start_x = start_x[None, :].expand(num_trials, -1)
        for j in range(num_steps):
            with torch.no_grad():
                start_x = self.eval_step_interactive(start_x, edit_data[j], edited_mask[j], extra_dict).detach()
            output_xs[:,j, :] = start_x
        return output_xs

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

    def compute_loss(self, last_x, next_x, ts, extra_dict):
        #[4096,1335]
        ### target reach
        # target_xz = extra_dict.get('target', None)
        # if target_xz is not None:
        #     last_x = torch.cat([last_x, target_xz], dim=-1)  # [B, cond_dim + 2]

        estimated, noise, xt, ts = self.diffusion(last_x, next_x, ts, extra_dict)
        # print("last_x", last_x.shape, next_x.shape)   # torch.Size([4096, 1602]) torch.Size([4096, 1602])
        if self.estimate_mode == 'x0':
            target = next_x
            pred_x0 = estimated

        elif self.estimate_mode == 'epsilon':
            target = noise
            pred_x0 = self.diffusion.get_x0_from_xt(xt, ts, estimated)
        
        else:
            assert(False), "Unsupported estimate mode: {}".format(self.estimate_mode) 

        if self.loss_type == 'l1':
            loss_diff = torch.nn.functional.l1_loss(estimated, target.squeeze())

        elif self.loss_type == 'l2':
            #loss_diff = torch.sum(torch.square(target - estimated), dim=-1).mean()
            loss_diff = torch.nn.functional.mse_loss(estimated, target.squeeze())
        # print("pred_x0,pred_x0.shape",pred_x0,pred_x0.shape)

        ### conditions
        dr = torch.zeros((pred_x0.shape[0], self.block_size, 1)).to(self.device)
        displacement = torch.zeros((pred_x0.shape[0], self.block_size, 2)).to(self.device)
        loss_TG = 0
        if self.conds_flag:
            traj_pose_gt = extra_dict.get("traj_pose", None)
            traj_trans_gt = extra_dict.get("traj_trans", None)

            for i in range(self.block_size):
                pose_t = pred_x0[:, int(i * self.frame_dim / self.block_size): int((i+1) * self.frame_dim / self.block_size)]
                pose_denorm = self.dataset.denorm_data(pose_t, device=pose_t.device)
                dr[:, i, :] = self.dataset.get_heading_dr(pose_denorm)[..., None]  # Δyaw
                root_vel = self.dataset.get_root_linear_planar_vel(pose_denorm).to(self.device)  # (vx, vz)
                # root_rotmat_up = self.dataset.get_rotation_matrix(torch.zeros(1)).to(self.device)
                # displacement = (root_rotmat_up * root_vel.unsqueeze(1)).sum(dim=2).to(dtype=torch.float32)
                displacement[:, i, :] = root_vel
            # pose_t = pred_x0[:, : int(self.frame_dim / self.block_size)]
            # pose_denorm = self.dataset.denorm_data(pose_t, device=pose_t.device)
            # dr = self.dataset.get_heading_dr(pose_denorm)[..., None]  # Δyaw
            # root_vel = self.dataset.get_root_linear_planar_vel(pose_denorm).to(self.device)  # (vx, vz)
            # # root_rotmat_up = self.dataset.get_rotation_matrix(torch.zeros(1)).to(self.device)
            # # displacement = (root_rotmat_up * root_vel.unsqueeze(1)).sum(dim=2).to(dtype=torch.float32)
            # displacement= root_vel
            dr = dr.reshape(pred_x0.shape[0], -1)     ### conditions_2
            displacement = displacement.reshape(pred_x0.shape[0], -1)

            # cond_scale = displacement.abs().max().detach() * 2 + 1e-6  # norm
            cond_scale = displacement.abs().max().detach() * 2 + 1e-6
            # print("max min", displacement.abs().max().detach(), displacement.abs().min().detach())
            traj_trans_gt = traj_trans_gt / cond_scale
            displacement = displacement / cond_scale

            loss_pose = 0
            loss_traj = 0
            # for i in range(self.block_size):
            loss_pose = f.l1_loss(dr, traj_pose_gt)
            loss_traj = f.mse_loss(displacement, traj_trans_gt)
            loss_TG = (loss_traj + loss_pose)     ### trans only

            ### conditions_2
            # loss_pose = f.l1_loss(dr, traj_pose_gt[:, :1])
            # loss_traj = f.mse_loss(displacement, traj_trans_gt[:, :2])
            # loss_TG = loss_traj + loss_pose

        # overlap
        prev, curr = last_x.reshape(last_x.shape[0], self.block_size, int(last_x.shape[1] / self.block_size)), pred_x0.reshape(last_x.shape[0], self.block_size, int(last_x.shape[1] / self.block_size))
        # print("prev", prev.shape, curr.shape)
        # (1) 位置不连续
        pos_err = 0
        # for i in range(self.overlap):
        pos_err += torch.norm(curr[:, 0, :] - prev[:, -1, :])
        # (2) 速度不连续
        # prev_vel = prev[:, -(self.overlap + 1), :] - prev[:, -(self.overlap + 2), :]
        # curr_vel = curr[:, 1, :] - curr[:, 0, :]
        # vel_err = torch.norm(curr_vel - prev_vel)
        total_err = pos_err
        # print("pos_err, vel_err", pos_err, vel_err)

        return loss_diff, pred_x0, total_err, loss_TG
    
    def get_model_params(self):
        params = list(self.diffusion.parameters())
        return params

    def update(self):
        if self.use_ema:
            self.update_ema()

    def update_ema(self):
        self.ema_step += 1
        if self.ema_step % self.ema_update_rate == 0:
            if self.ema_step < self.ema_start:
                self.ema_diffusion.load_state_dict(self.diffusion.state_dict())
            else:
                self.ema.update_model_average(self.ema_diffusion, self.diffusion)


class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    """
    def __init__(
        self,
        config
    ):
        super().__init__()

        self.T = config["diffusion"]['T']
        self.schedule_mode = config["diffusion"]["noise_schedule_mode"]
        self.estimate_mode = config["diffusion"]["estimate_mode"]
        self.norm_type = config["model_hyperparam"]["norm_type"]
        self.act_type = config["model_hyperparam"]["act_type"]
        self.time_emb_dim = config["model_hyperparam"]["time_emb_size"]
        self.hidden_dim = config["model_hyperparam"]["hidden_size"]
        self.layer_num = config["model_hyperparam"]["layer_num"]
        self.frame_dim = config['frame_dim']

        self.block_size = config['block']['size']
        self.conds_flag = config["block"]["conds_flag"]

        self.model = NoiseDecoder(self.frame_dim, self.hidden_dim, self.time_emb_dim, self.layer_num, self.norm_type, self.act_type, self.conds_flag)
        self.time_mlp = torch.nn.Sequential(
            Embedding.PositionalEmbedding(self.time_emb_dim, 1.0),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
            Activation.SiLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        if self.conds_flag:
            self.cond_emb_dim = self.time_emb_dim
            self.cond_proj = nn.Linear(3 * self.block_size, self.cond_emb_dim)  ### conditions  # conditions_2 3 * self.block_size
        
        betas = self._generate_diffusion_schedule()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1. / alphas)))
        self.register_buffer("reciprocal_sqrt_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas_cumprod_m1", to_torch(np.sqrt(1. / alphas_cumprod -1)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))


    def _generate_diffusion_schedule(self, s=0.008):
        def f(t, T):
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
        
        if self.schedule_mode == 'cosine':  
            # from https://arxiv.org/abs/2102.09672  
            alphas = []
            f0 = f(0, self.T)

            for t in range(self.T + 1):
                alphas.append(f(t, self.T) / f0)
            
            betas = []

            for t in range(1, self.T + 1):
                betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
            return np.array(betas)
        
        elif self.schedule_mode == 'uniform':
            # from original ddpm paper
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start, beta_end, self.T)
        
        elif self.schedule_mode == 'quadratic':
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start**0.5, beta_end**0.5, self.T) ** 2
        
        elif self.schedule_mode == 'sigmoid':
            beta_start = 0.0001
            beta_end = 0.02
            betas = np.linspace(-6, 6, self.T)
            return 1/(1+np.exp(-betas)) * (beta_end - beta_start) + beta_start
        
        else:
            assert(False), "Unsupported diffusion schedule: {}".format(self.schedule_mode)
    

    @torch.no_grad()
    def extract(self, a, ts, x_shape):
        b, *_ = ts.shape
        out = a.gather(-1, ts)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    
    @torch.no_grad()
    def add_noise(self, x, ts):
        return x + self.extract(self.sigma, ts, x.shape) * torch.randn_like(x)
    
    def add_noise_w(self, x, ts, noise):
        return x + self.extract(self.sigma, ts, x.shape) * noise#torch.randn_like(x)

    @torch.no_grad()
    def compute_alpha(self, beta, ts):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, ts + 1).view(-1, 1)
        return a
    

    @torch.no_grad()
    def remove_noise(self, xt, pred, ts):
        output =  (xt - self.extract(self.remove_noise_coeff, ts, pred.shape) * pred) * \
                self.extract(self.reciprocal_sqrt_alphas, ts, pred.shape)
        
        return output
    
    def get_x0_from_xt(self, xt, ts, noise):    ###从带噪声的xt估计原始x0
        output =  (xt - self.extract(self.sqrt_one_minus_alphas_cumprod, ts, xt.shape) * noise) * \
                self.extract(self.reciprocal_sqrt_alphas_cumprod, ts, xt.shape)
        return output

    def get_eps_from_x0(self, xt, ts, pred_x0):     ###从x0估计噪声ε
        return (xt * self.extract(self.reciprocal_sqrt_alphas_cumprod, ts, xt.shape)  - pred_x0) / \
            self.extract(self.reciprocal_sqrt_alphas_cumprod_m1, ts, xt.shape)


    def perturb_x(self, x, ts, noise):
        return (
            self.extract(self.sqrt_alphas_cumprod, ts, x.shape) * x +
            self.extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape) * noise
        )   

    def corr_reward(self, x_gen, x_in):

        # 在 frame_dim 上计算每一帧的皮尔森相关
        x_gen_centered = x_gen - x_gen.mean(dim=2, keepdim=True)
        x_in_centered = x_in - x_in.mean(dim=2, keepdim=True)
        numerator = (x_gen_centered * x_in_centered).sum(dim=2)
        denominator = torch.sqrt((x_gen_centered ** 2).sum(dim=2) * (x_in_centered ** 2).sum(dim=2) + 1e-8)
        corr_per_frame = numerator / denominator
        # 求整个块的平均
        return corr_per_frame.mean()

    def sample_ddpm_best_of_n(self, last_x, extra_info, N=10, reward_fn=None):
        """
        N: number of parallel samples
        reward_fn: function(x) -> scalar reward
        """
        all_samples = []
        all_rewards = []

        B, total_dim = last_x.shape
        T = self.block_size
        D = total_dim // T
        for i in range(N):
            x_gen = self.sample_ddpm(last_x, extra_info, record_process=False)
            all_samples.append(x_gen)

            x_gen = x_gen.view(B, T, D)
            x_in = last_x.view(B, T, D)
            x_gen_diff = x_gen[:, 1:, :] - x_gen[:, :-1, :]
            x_in_diff = x_in[:, 1:, :] - x_in[:, :-1, :]
            r = self.corr_reward(x_gen_diff, x_in_diff)

            all_rewards.append(r)

        all_rewards = torch.tensor(all_rewards)
        best_idx = torch.argmax(all_rewards)
        best_sample = all_samples[best_idx]

        return best_sample

    @torch.no_grad()
    def sample_ddpm(self, last_x, extra_info, record_process=False):
        ### 迭代去噪过程，从随机噪声开始，通过T个时间步逐步去除噪声生成数据
        x = torch.randn(last_x.shape[0], last_x.shape[-1]).to(last_x.device)
        #ce = None if self.use_cond else self.cond_mlp(extra_info['cond'])

        # ###
        # if last_x.shape[-1] == self.frame_dim:
        #     target = extra_info.get('target', None)
        # # === 拼接 target 作为条件 ===
        # if target is not None:
        #     last_x = torch.cat([last_x, target], dim=-1)  # [B, cond_dim + 2]

        if record_process:
            x0s = torch.zeros(last_x.shape[0], self.T, last_x.shape[-1], device=last_x.device)

        ### conditions
        if self.conds_flag:
            traj_pose, traj_trans = extra_info["traj_pose"].to(last_x.device), extra_info["traj_trans"].to(
                last_x.device)
            # print(traj_pose.shape, traj_trans.shape)

            cond = torch.cat([traj_pose, traj_trans], dim=-1)
            # cond = traj_trans
            cond_emb = self.cond_proj(cond)

        for t in range(self.T - 1, -1, -1):
            ts = torch.tensor([t], device = last_x.device).repeat(last_x.shape[0])
            te = self.time_mlp(ts)

            if self.conds_flag:
                latent = torch.cat((te, cond_emb), dim=-1)  ### conditions
                pred = self.model(last_x, x, latent).detach()  ### conditions te ### latent
            else:
                pred = self.model(last_x, x, te).detach()
            
            ###
            if torch.isnan(pred).any():
                print(f"[NaN] step , pred has NaN")
            # else:
                # print(f"pred max: {pred.max()}, min: {pred.min()}")
            # print(f"te : {te} ")
            # nan_count = torch.isnan(pred).sum().item()
            # total = pred.numel()
            # print(f"pred has {nan_count} NaNs out of {total} elements")
            
            if self.estimate_mode == 'epsilon':
                x = self.remove_noise(x, pred, ts)
            elif self.estimate_mode == 'x0':
                x = pred
            if record_process:
                x0s[:,self.T - 1- t,:] = x

            # # === 基于 target 的引导 ===     ###
            # if target is not None:
            #     # 计算方向偏置（假设 root 在 self.root_xz）
            #     direction = F.normalize(target - self.root_xz, dim=-1)  # (B,2)
            #     guide_strength = 0.2  # 可调系数
            #     # 将偏置加到输出的 root velocity 部分（前 2 维）
            #     x[:, :2] += guide_strength * direction
                
            if t > 0:
                x = self.add_noise(x, ts)
        
        if record_process:
            return x0s
        
        return x
    
    def sample_rl_ddpm(self, last_x, action_dict, extra_info):
        
        steps = extra_info['action_step']
        train_rand_scale = extra_info['rand_scale']
        test_rand_scale = extra_info['test_rand_scale']
        clip_scale = extra_info['clip_scale']
        guidance_scale = 0.15

        action_mode = extra_info['action_mode']
        is_train = extra_info['is_train']
        
        action_scale = extra_info['action_scale'] if is_train else extra_info['test_action_scale']

        # action_dim_per_step = 8 if action_mode == 'loco' else int(self.frame_dim / self.block_size)  ### per frame

        # ### action block * 3(xz dr)
        # # print("action_dict", action_dict[122])
        # action = action_dict.view(-1, self.block_size, 3)
        # batch_size = last_x.shape[0]
        # # 2. 初始化纯噪声 x_T，准备生成下一个动作块
        # x_t = torch.randn(batch_size, self.frame_dim, device=last_x.device)
        # for t in range(self.T - 1, -1, -1):
        #     # 构造当前时间步张量
        #     ts = torch.tensor([t], device=last_x.device).repeat(last_x.shape[0])
        #     te = self.time_mlp(ts)
        #     # 3. 核心前向预测：传入上一块状态 xcur、当前噪声态 x_t 和时间/条件 latent
        #     pred = self.model(last_x, x_t, te).detach()
        #     # 4. 推导当前的干净数据预测值 (pred_x0)
        #     # 扩散模型不管预测的是什么（噪声 eps 或数据 x0），最终都要用来还原 x0
        #     if self.estimate_mode == 'epsilon':
        #         x = self.remove_noise(x_t, pred, ts)
        #         print("estimate_mode == epsilon")
        #     elif self.estimate_mode == 'x0':
        #         x = pred.view(batch_size, self.block_size, int(self.frame_dim / self.block_size))
        #     dr_dim = 3  # lafan：2
        #     guided_x0 = x.clone()
        #     current_weight = guidance_scale * (t / self.T)
        #     # print("guided_x0", guided_x0.shape, dr_dim)
        #     if t > 1:
        #         # guided_x0[:, :, :dr_dim] = x[:, :, :dr_dim] + current_weight * (action - x[:, :, :dr_dim])
        #         guided_x0[:, :, :dr_dim] = x[:, :, :dr_dim] + current_weight * action
        #
        #     x = guided_x0.view(batch_size, -1)
        #     if t > 0:
        #         x = self.add_noise(x, ts)
        #     # else:
        #     #     print("x:", x[0][:3])

        ### action block_size * frame_dim
        batch_size = last_x.shape[0]
        action_dim_per_step = 8 if action_mode == 'loco' else self.frame_dim
        single_dim = int(self.frame_dim / self.block_size)
        # 1. 带有 RL 偏置的初始噪声 (RL-Biased Initialization)
        # 取 Actor 输出的第 0 个特征块作为初始引导偏置
        init_bias = action_dict[..., :action_dim_per_step] / 3.0
        # 构造纯随机噪声 x_T
        x = torch.randn(batch_size, self.frame_dim, device=last_x.device)

        # 将 RL 偏置注入初始噪声中
        x += init_bias

        # 2. 步进式去噪循环
        for t in range(self.T - 1, -1, -1):
            with torch.no_grad():
                ts = torch.tensor([t], device=last_x.device).repeat(batch_size)
                te = self.time_mlp(ts)

                # 预测去噪均值
                pred = self.model(last_x, x, te).detach()

                if self.estimate_mode == 'epsilon':
                    x = self.remove_noise(x, pred, ts)
                elif self.estimate_mode == 'x0':
                    x = pred

            # 3. 强化学习步进式暴力干预 (Step-wise RL Intervention)
            if t in steps:
                # 注意：索引 i 需要 +1，因为 index 0 被用作初始偏置了
                i = steps.index(t) + 1

                dx = action_dict[..., i * action_dim_per_step: (i + 1) * action_dim_per_step]

                energy_scale = 1.0 / (self.block_size ** 0.5)
                dx = dx * energy_scale

                dx_blocks = dx.view(batch_size, self.block_size, -1)
                if self.block_size > 2:
                    smoothed_dx = dx_blocks.clone()
                    # 用前后帧的均值平滑中间帧
                    smoothed_dx[:, 1:-1, :] = 0.5 * dx_blocks[:, 1:-1, :] + \
                                              0.25 * dx_blocks[:, :-2, :] + \
                                              0.25 * dx_blocks[:, 2:, :]
                    # 重新展平回 [Batch, 1335]
                    dx = smoothed_dx.view(batch_size, -1)

                rand_scale = train_rand_scale if is_train else test_rand_scale
                noise = torch.randn_like(dx)
                sigma_val = self.extract(self.sigma, ts, x.shape)[0]

                # 将 Actor 输出的整块残差 (1335维) 直接叠加
                x += action_scale * (dx + rand_scale * noise * sigma_val)

                # 安全截断，防止大动作崩坏流形
                x = torch.clamp(x, -clip_scale, clip_scale)

            # 4. 加噪流转到下一步
            if t > 0:
                x = self.add_noise(x, ts)

        ###
        # x = action_dict[..., :action_dim_per_step] / 3  ### per frame
        # x0 = x
        # for i in range(self.block_size - 1):
        #     x = torch.cat([x, x0], dim=-1)
        #
        # for t in range(self.T - 1, -1, -1): # 7-0
        #     with torch.no_grad():
        #
        #         ts = torch.tensor([t], device=last_x.device).repeat(last_x.shape[0])
        #         te = self.time_mlp(ts)
        #         pred = self.model(last_x, x, te).detach()
        #
        #
        #         if self.estimate_mode == 'epsilon':
        #             x = self.remove_noise(x, pred, ts)
        #         elif self.estimate_mode == 'x0':
        #             x = pred
        #
        #     if t in steps:
        #         i = steps.index(t) + 1
        #         dx = action_dict[..., i*action_dim_per_step:(i+1)*action_dim_per_step]
        #         rand_scale = train_rand_scale if is_train else test_rand_scale
        #         # print("rand_scale", rand_scale)
        #
        #         rand_scale *= torch.randn_like(dx)
        #
        #         ### per frame
        #         for i in range(self.block_size):
        #             weight = 1 - i / self.block_size  # 线性衰减，例如 [1.0, 0.8, 0.6, 0.4, 0.2]
        #             x[:, i * action_dim_per_step:(i + 1) * action_dim_per_step] += \
        #                 weight * action_scale * (dx + rand_scale * self.extract(self.sigma, ts, x.shape)[0])
        #         # x[:, 0:action_dim_per_step] += action_scale * (dx + rand_scale * self.extract(self.sigma, ts, x.shape)[0])
        #
        #         # x += action_scale * (dx + rand_scale * self.extract(self.sigma, ts, x.shape)[0])
        #         x = torch.clamp(x, -clip_scale, clip_scale)
        #
        #     if t > 0:
        #         x = self.add_noise(x, ts)
        return x

    

    @torch.no_grad()
    def sample_ddpm_interactive(self, last_x, edited_mask, edited_data, extra_info):
        repaint_step = extra_info['repaint_step']
        interact_stop_step = extra_info['interact_stop_step']
        edited_mask_inv = 1 - edited_mask

        x = torch.randn(last_x.shape[0], last_x.shape[-1]).to(last_x.device)
        
        for t in range(self.T - 1, -1, -1):
            for t_rp in range(repaint_step):
                ts = torch.tensor([t], device = last_x.device).repeat(last_x.shape[0])

                te = self.time_mlp(ts)
                pred = self.model(last_x, x, te).detach()
                
                if self.estimate_mode == 'epsilon':
                    x = self.remove_noise(x, pred, ts)
                elif self.estimate_mode == 'x0':
                    x = pred
                
                cur_edited_mask_inv = edited_mask_inv.clone()
                if t > interact_stop_step:
                    #cur_edited_mask_inv = torch.randn_like(edited_mask_inv)
                    x = edited_data * edited_mask + x * cur_edited_mask_inv #x* cur_edited_mask_inv
               
                if t > 0:
                    #if t_rp < repaint_step and t != self.T-1 and t > interact_stop_step:
                    #    ts = torch.tensor([t+1], device = last_x.device).repeat(last_x.shape[0])
                    x = self.add_noise(x, ts)

        return x
    


    @torch.no_grad()
    def sample_ddim(self, bs, num_steps, device, eta=0.0):
        
        T_train = self.T - 1
        timesteps = torch.linspace(0, T_train, num_steps, dtype=torch.long, device = device)
        timesteps_next = [-1] + list(timesteps[:-1])
        
        x = torch.randn(bs, self.action_dim, device = device)
        
        for t in range(len(timesteps)-1, -1, -1):
            
            ts = torch.tensor([timesteps[t]], device = device).repeat(bs)
            ts1 = torch.tensor([timesteps_next[t]], device = device).repeat(bs)
            
            alpha_bar = self.extract(self.alphas_cumprod, ts, x.shape)
            alpha_bar_prev = self.extract(self.alphas_cumprod, ts1, x.shape) if t > 0 else torch.ones_like(x)
            sigma = eta *((1 - alpha_bar_prev)/(1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)).sqrt()
            # eta = 0.0 deterministic
            te = self.time_mlp(ts)
            input = torch.cat([x, te],axis=-1)
            output = self.model(input)

            if self.estimate_mode == 'x0':
                pred_x0 = output
                pred_eps = self.get_eps_from_x0(x, ts, pred_x0)
            else:
                pred_eps = output
                pred_x0 = self.get_x0_from_xt(x, ts, pred_eps)

            mean_pred = (
                pred_x0 * self.extract(self.sqrt_alphas_cumprod, ts, x.shape)
                + (1 - alpha_bar_prev- sigma ** 2).sqrt() * pred_eps
            )

            nonzero_mask = (
                (ts != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  
            
            x = mean_pred + nonzero_mask * sigma * torch.randn_like(x)            
    
        return x
    
    def forward(self, cur_x, next_x, ts, extra_info):
        ### conditions
        if self.conds_flag:
            traj_pose, traj_trans = extra_info["traj_pose"], extra_info["traj_trans"]
            cond = torch.cat([traj_pose, traj_trans], dim=-1)
            # cond = traj_trans
            # print("cond", cond.shape)
            cond_emb = self.cond_proj(cond)   ### trans only


        # print("cur_x,next_x",cur_x.shape,next_x.shape)
        bs = cur_x.shape[0]
        device = cur_x.device
        if ts is None:
            ts = torch.randint(0, self.T, (bs,), device=device)
        
        time_emb = self.time_mlp(ts) 

        noise = torch.randn_like(next_x)
        perturbed_x = self.perturb_x(next_x, ts.clone(), noise) #xt（加噪后的下一帧数据）

        if self.conds_flag:
            latent = torch.cat([time_emb, cond_emb], dim=-1)  ### time_emb
        else:
            latent = time_emb
        estimated = self.model(cur_x, perturbed_x, latent)  ### latent
        return estimated, noise, perturbed_x, ts


class NoiseDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        hidden_size,
        time_emb_size,
        layer_num,
        norm_type,
        act_type,
        conds_flag
    ):
        super().__init__()

        conds_emb_size = 0
        if conds_flag:
            conds_emb_size = time_emb_size

        self.input_size = frame_size
        layers = []
        for _ in range(layer_num): 
            if act_type == 'ReLU':
                non_linear = torch.nn.ReLU() ### v12 is ReLU
            elif act_type == 'SiLU':
                non_linear = Activation.SiLU() 
            linear = nn.Linear(hidden_size + frame_size * 2 + time_emb_size + conds_emb_size, hidden_size)   ### conditions * 2
            if norm_type == 'layer_norm':
                norm_layer = nn.LayerNorm(hidden_size)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(16, hidden_size)  ###16

            layers.append(norm_layer)
            layers.extend([non_linear, linear])
            
        self.net = nn.ModuleList(layers)
        self.fin = nn.Linear(frame_size * 2 + time_emb_size + conds_emb_size, hidden_size)   ### conditions
        self.fco = nn.Linear(hidden_size + frame_size * 2 + time_emb_size + conds_emb_size, frame_size)  ### conditions
        self.act = Activation.SiLU()
  
    def forward(self, xcur, xnext, latent):
        
        x0 = xnext
        y0 = xcur
        
        x = torch.cat([xcur, xnext, latent], dim=-1)
        x = self.fin(x)

        for i, layer in enumerate(self.net):
            if i % 3 == 2:
                ###
                # print(f"x max: {x.max()}, min: {x.min()}")

                x = torch.cat([x, x0, y0, latent], dim=-1)
                x = layer(x)
            else:
                x = layer(x)

        x = torch.cat([x, x0, y0, latent],dim=-1) 
        x = self.fco(x)
        return x 
   