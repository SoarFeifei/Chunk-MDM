import abc
import copy
import numpy as np

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

import util.vis_util as vis_util
import util.logging as logging_util
import util.save as save_util

import util.eval as eval_util   # eval

import yaml



class BaseTrainer():
    def __init__(self, config, dataset, device):
        self.config = config
        self.device = device
        self.dataset = dataset

        ###
        self.block_size = config['block']['size']
        self.conds_flag = config['block']['conds_flag']

        optimizer_config = config['optimizer']
        self.batch_size = optimizer_config['mini_batch_size']
        self.num_rollout = optimizer_config['rollout']
        self.initial_lr = optimizer_config['initial_lr']
        self.final_lr = optimizer_config['final_lr']
        self.peak_student_rate = optimizer_config.get('peak_student_rate', 1.0)
        self._get_schedule_samp_routines(config['optimizer'])

        test_config = config['test']
        self.test_interval = test_config["test_interval"]
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]

        self.frame_dim = int(dataset.frame_dim / self.block_size)
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)

        self.logger = logging_util.wandbLogger(proj_name="{}_{}".format(self.NAME, dataset.NAME), run_name=self.NAME)

        self.plot_jnts_fn = self.dataset.plot_jnts if hasattr(self.dataset, 'plot_jnts') and callable(
            self.dataset.plot_jnts) \
            else vis_util.vis_skel

        self.plot_traj_fn = self.dataset.plot_traj if hasattr(self.dataset, 'plot_traj') and callable(
            self.dataset.plot_traj) \
            else vis_util.vis_traj
        return

    @abc.abstractmethod
    def train_loop(self, model):
        return

    def _init_optimizer(self, model):
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.initial_lr)

    def _update_lr_schedule(self, optimizer, epoch):
        """Decreases the learning rate linearly"""
        lr = self.initial_lr - (self.initial_lr - self.final_lr) * epoch / float(self.total_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _get_schedule_samp_routines(self, optimizer_config):
        self.anneal_times = optimizer_config['anneal_times']
        self.initial_teacher_epochs = optimizer_config.get('initial_teacher_epochs', 1)
        self.end_teacher_epochs = optimizer_config.get('end_teacher_epochs', 1)
        self.teacher_epochs = optimizer_config['teacher_epochs']
        self.ramping_epochs = optimizer_config['ramping_epochs']
        self.student_epochs = optimizer_config['student_epochs']
        self.use_schedule_samp = self.ramping_epochs != 0 or self.student_epochs != 0

        self.initial_schedule = torch.zeros(self.initial_teacher_epochs)
        self.end_schedule = torch.zeros(self.end_teacher_epochs)
        self.sample_schedule = torch.cat([
            # First part is pure teacher forcing
            torch.zeros(self.teacher_epochs),
            # Second part with schedule sampling
            torch.linspace(0.0, self.peak_student_rate, self.ramping_epochs),
            # last part is pure student
            torch.ones(self.student_epochs) * self.peak_student_rate,

        ])
        self.sample_schedule = torch.cat([self.sample_schedule for _ in range(self.anneal_times)], axis=-1)
        self.sample_schedule = torch.cat([self.initial_schedule, self.sample_schedule, self.end_schedule])

        self.total_epochs = self.sample_schedule.shape[0]
        print("self.sample_schedule.shape[0]", self.sample_schedule.shape)

    def train_model(self, model, out_model_file, int_output_dir, log_file):
        self._init_optimizer(model)
        for ep in range(self.total_epochs):
            loss_stats = self.train_loop(ep, model)
            if ep == 0:
                continue
            if ep % self.test_interval == 0:
                num_nans = self.evaluate(ep, model, int_output_dir)
                save_util.save_weight(model, int_output_dir + '_ep{}.pth'.format(ep))
                save_util.save_weight(model, out_model_file)

            self.logger.log_epoch(loss_stats)
            self.logger.print_log(loss_stats)

        save_util.save_weight(model, out_model_file)

    ### conditions
    def test_conditional_rollout(self, device, total_frames=120, block_size=1):
        # ===== 生成一条连续轨迹 =====
        t = torch.linspace(0, 2 * np.pi, total_frames)
        radius = 5.0
        traj_x = radius * torch.cos(t)
        traj_z = radius * torch.sin(t)
        traj_trans = torch.stack([traj_x, traj_z], dim=-1).to(device)
        traj_pose = torch.atan2(traj_z, traj_x).unsqueeze(-1)

        # ===== 逐块生成 =====
        n_blocks = total_frames // block_size
        traj_trans_accum = torch.zeros(n_blocks - 1, block_size, 2)  # conditions_2 [n_blocks-1, 2] conditions n_blocks - 1, block_size, 2
        traj_pose_accum = torch.zeros(n_blocks - 1, block_size, 1)  # [n_blocks-1, 1]
        for b in range(n_blocks - 1):
            start = b * block_size
            end = (b + 1) * block_size
            for i in range(block_size):
                traj_trans_accum[b, :] = traj_trans[b + 1] - traj_trans[b]  ### conditions_2
                traj_trans_local = traj_trans[b + 1] - traj_trans[b]
                traj_trans_accum[b] = traj_trans_local
                # traj_pose_accum[b] = traj_pose[b]   ###
        traj_trans_accum = traj_trans_accum.reshape(n_blocks - 1, -1)
        traj_pose_accum = traj_pose_accum.reshape(n_blocks - 1, -1)
        conds = {
            "traj_trans": traj_trans_accum,  # [1, 2]
            "traj_pose": traj_pose_accum,  # [1, 1]
        }

        return conds
    def evaluate(self, ep, model, result_ouput_dir):
        model.eval()
        NaN_clip_num = 0

        ### conditions
        if self.conds_flag:
            st_idx = 1
            ref_clip = self.dataset.motion_flattened[0:int(120 / self.block_size)+1]
            # print("ref_clip", ref_clip.shape)
            # ref_clip = ref_clip[0]
            test_out_lst = []
            test_local_out_lst = []
            start_x = torch.from_numpy(ref_clip[0, :]).float().to(self.device)
            start_x = start_x.reshape(1, -1)
            displacement = torch.zeros(int(120 / self.block_size), self.block_size, 2)
            dr = torch.zeros(int(120 / self.block_size), self.block_size, 1)
            # displacement = torch.zeros(20, 2)
            # dr = torch.zeros(20, 1)
            for i in range(int(120 / self.block_size)):
                future_pose = ref_clip[i + 1, :]
                future_pose = torch.from_numpy(future_pose.reshape(1, -1))
                # print("future_pose", future_pose.shape)
                for j in range(self.block_size):  ### conditions_2
                    frame_pose = future_pose[:, self.frame_dim * j:self.frame_dim * (j+1)]
                    pose_denorm = self.dataset.denorm_data(frame_pose)
                    root_xz_vel = self.dataset.get_root_linear_planar_vel(pose_denorm)
                    dr[i, j, :] = self.dataset.get_heading_dr(pose_denorm)[..., None].to(dtype=torch.float32)
                    displacement[i, j, :] = root_xz_vel
                # frame_pose = future_pose[:, :self.frame_dim]
                # pose_denorm = self.dataset.denorm_data(frame_pose)
                # root_xz_vel = self.dataset.get_root_linear_planar_vel(pose_denorm)
                # dr[i, :] = self.dataset.get_heading_dr(pose_denorm)[..., None].to(dtype=torch.float32)
                # displacement[i, :] = root_xz_vel

            dr = dr.reshape(int(120 / self.block_size), -1)
            displacement = displacement.reshape(int(120 / self.block_size), -1)

            if ep == 0:
                model_lst = self.dataset.data_component
                cur_jnts = []
                for mode in model_lst:
                    jnts_mode = self.dataset.x_to_jnts(self.dataset.denorm_data(ref_clip), mode=mode)
                    cur_jnts.append(jnts_mode)
                cur_jnts = np.array(cur_jnts)

                self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/gt_{}'.format(st_idx))
                ref_clip = cur_jnts[[0], ...]
            else:
                ref_clip = ref_clip.reshape(ref_clip.shape[0] * self.block_size, self.frame_dim)
                # denorm_ref_clip = self.dataset.denorm_data(ref_clip).reshape(int(ref_clip.shape[0] / self.block_size),self.frame_dim * self.block_size)
                denorm_ref_clip = self.dataset.denorm_data(ref_clip)
                ref_clip = self.dataset.x_to_jnts(denorm_ref_clip, mode=self.dataset.data_component[0])[
                    None, ...]

            conds = {}  ### conditions
            conds["traj_trans"] = displacement.to(self.device)
            conds["traj_pose"] = dr.to(self.device)
            # conds_long = self.test_conditional_rollout(self.device, 1001)  ### conditions

            ref_clip = ref_clip[:, 0:90, :, :]
            # print("ref_clip", ref_clip.shape)
            test_out_lst.append(ref_clip.squeeze())
            test_data = model.eval_seq(start_x, conds, 90, self.test_num_trials)
            # test_data_long = model.eval_seq(start_x, conds_long, 1000, 3)

            num_all = torch.numel(test_data)
            num_nans = torch.sum(torch.isnan(test_data))

            # num_all_long = torch.numel(test_data_long)
            # num_nans_long = torch.sum(torch.isnan(test_data_long))

            print('percent of nan frames : {}'.format(num_nans * 1.0 / num_all))
            # print('percent of nan frames for long horizon gen : {}'.format(num_nans_long*1.0/num_all_long))
            should_plot = True
            if num_nans > 0:
                NaN_clip_num += 1
                should_plot = False
                # print('skip calc stats {} to save time'.format(st_idx))
                # if False:#NaN_clip_num >= len(self.dataset.test_valid_idx)-1:
                # continue # skip calc stats to save time
            test_data = test_data.detach().cpu().numpy()
            # print("test_data", test_data.shape)
            for i in range(test_data.shape[0]):
                cur_denormed_test_data = self.dataset.denorm_data(copy.deepcopy(test_data[i]))
                cur_jnts = []

                for mode in self.dataset.data_component:
                    jnts_mode = self.dataset.x_to_jnts(cur_denormed_test_data, mode=mode)
                    cur_jnts.append(jnts_mode)

                    if mode == self.dataset.data_component[0]:
                        test_out_lst.append(jnts_mode)
                        jnts_mode_local = jnts_mode - jnts_mode[:, [0], :]
                        test_local_out_lst.append(jnts_mode_local)
                cur_jnts = np.array(cur_jnts)
                if should_plot:
                    self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/{}_{}'.format(st_idx, i))
            test_out_lst = np.array(test_out_lst)
            print("test_out_lst", test_out_lst.shape)
            self.plot_traj_fn(test_out_lst, result_ouput_dir + '/{}'.format(st_idx))

        apd_mean = 0
        ade_mean = 0
        fde_mean = 0
        for idx, (st_idx, ref_clip) in enumerate(zip(self.dataset.test_valid_idx, self.dataset.test_ref_clips)):
            print('Eval Index:', st_idx)
            test_out_lst = []
            test_local_out_lst = []

            print("ref_clip", ref_clip.shape)       # ref_clip (12, 1335)
            start_x = torch.from_numpy(ref_clip[0]).float().to(self.device)
            print("start_x", start_x.shape)
            if ep == 0:
                model_lst = self.dataset.data_component
                cur_jnts = []
                ###
                ref_clip = ref_clip.reshape(ref_clip.shape[0] * self.block_size, self.frame_dim)
                # denorm_ref_clip = self.dataset.denorm_data(ref_clip).reshape(int(ref_clip.shape[0] / self.block_size),self.frame_dim * self.block_size)
                denorm_ref_clip = self.dataset.denorm_data(ref_clip)
                for mode in model_lst:

                    jnts_mode = self.dataset.x_to_jnts(denorm_ref_clip, mode=mode)
                    cur_jnts.append(jnts_mode)
                cur_jnts = np.array(cur_jnts)

                self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/gt_{}'.format(st_idx))
                ref_clip = cur_jnts[[0], ...]
            else:
                ###
                ref_clip = ref_clip.reshape(ref_clip.shape[0] * self.block_size, self.frame_dim)
                # print("ref_clip", ref_clip.shape)   # ref_clip (60, 267)
                # denorm_ref_clip = self.dataset.denorm_data(ref_clip).reshape(int(ref_clip.shape[0] / self.block_size),self.frame_dim * self.block_size)
                denorm_ref_clip = self.dataset.denorm_data(ref_clip)
                # print("denorm_ref_clip", denorm_ref_clip.shape)
                ref_clip = self.dataset.x_to_jnts(denorm_ref_clip, mode=self.dataset.data_component[0])[None, ...]

            conds = None
            if self.conds_flag:
                conds = self.test_conditional_rollout(self.device, block_size=self.block_size)     ### conditions

            ref_local_clip = ref_clip - ref_clip[:, :, [0], :]

            ref_clip_0 = ref_clip   # eval

            test_out_lst.append(ref_clip.squeeze())
            # print("test_out_lst", np.array(test_out_lst).shape)     # (1, 56, 22, 3)
            test_data = model.eval_seq(start_x, conds, np.array(test_out_lst).shape[1], self.test_num_trials)    # overlap  self.test_num_steps

            # test_data_long = model.eval_seq(start_x, None, 1000, 3)
            # print("test_data", test_data.shape)     # ([1, 60, 267])
            # print("test_data_long", test_data_long.shape)   # ([3, 1000, 267])

            # boundary error
            pos_err = 0
            vel_err = 0
            pred = test_data[0, :, :]
            # print("pred", pred.shape)
            for i in range(int(pred.shape[0]/self.block_size - 1)):
                prev, curr = pred[i * self.block_size:(i+1) * self.block_size, :], pred[(i+1) * self.block_size:(i+2) * self.block_size, :]
            # print("prev", prev.shape, curr.shape)
            # (1) 位置不连续
                pos_err += torch.norm(curr[0, :] - prev[-1, :])
            # # (2) 速度不连续
            #     prev_vel = prev[-1, :] - prev[-2, :]
            #     curr_vel = curr[1, :] - curr[0, :]
            #     vel_err += torch.norm(curr_vel - prev_vel)
            print("pos_err", pos_err)
            # print("vel_err", vel_err)

            # pos_err_long = 0
            # vel_err_long = 0
            # for j in range(int(test_data_long.shape[0])):
            #     pred_long = test_data_long[j, :, :]
            #     # print("pred", pred.shape)
            #     for i in range(int(pred_long.shape[0] / self.block_size - 1)):
            #         prev, curr = pred_long[i * self.block_size:(i + 1) * self.block_size, :], pred_long[(i + 1) * self.block_size:(i + 2) * self.block_size, :]
            #         # print("prev", prev.shape, curr.shape)
            #         # (1) 位置不连续
            #         pos_err_long += torch.norm(curr[0, :] - prev[-1, :])
            #         # # (2) 速度不连续
            #         # prev_vel = prev[-1, :] - prev[-2, :]
            #         # curr_vel = curr[1, :] - curr[0, :]
            #         # vel_err_long += torch.norm(curr_vel - prev_vel)
            # print("pos_err_long", pos_err_long / 3)
            # print("vel_err_long", vel_err_long / 3)

            num_all = torch.numel(test_data)
            num_nans = torch.sum(torch.isnan(test_data))

            # num_all_long = torch.numel(test_data_long)
            # num_nans_long = torch.sum(torch.isnan(test_data_long))

            print('percent of nan frames : {}'.format(num_nans * 1.0 / num_all))
            # print('percent of nan frames for long horizon gen : {}'.format(num_nans_long * 1.0 / num_all_long))
            should_plot = True
            if num_nans > 0:
                NaN_clip_num += 1
                should_plot = False
                # print('skip calc stats {} to save time'.format(st_idx))
                # if False:#NaN_clip_num >= len(self.dataset.test_valid_idx)-1:
                # continue # skip calc stats to save time

            test_data = test_data.detach().cpu().numpy()
            for i in range(test_data.shape[0]):
                ###
                # tmp_test_data = copy.deepcopy(test_data[i]).reshape(copy.deepcopy(test_data[i]).shape[0] * self.block_size,self.frame_dim)
                # denorm_tmp_test_data = self.dataset.denorm_data(tmp_test_data).reshape(int(tmp_test_data.shape[0] / self.block_size),self.frame_dim * self.block_size)
                denorm_tmp_test_data = self.dataset.denorm_data(copy.deepcopy(test_data[i]))
                # print("denorm_tmp_test_data", denorm_tmp_test_data.shape)

                cur_jnts = []

                for mode in self.dataset.data_component:
                    jnts_mode = self.dataset.x_to_jnts(denorm_tmp_test_data, mode=mode)
                    cur_jnts.append(jnts_mode)
                    # print("jnts_mode", jnts_mode.shape)
                    # print("mode", mode, self.dataset.data_component[0])

                    if mode == self.dataset.data_component[0]:
                        test_out_lst.append(jnts_mode)
                        jnts_mode_local = jnts_mode - jnts_mode[:, [0], :]
                        test_local_out_lst.append(jnts_mode_local)
                cur_jnts = np.array(cur_jnts)
                if should_plot:
                    self.plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/{}_{}'.format(st_idx, i))
            test_out_lst = np.array(test_out_lst)
            # print("ref_clip_0", ref_clip_0.shape)
            # print("test_out_lst", test_out_lst.shape)   # test_out_lst (2, 60, 22, 3)
            self.plot_traj_fn(test_out_lst, result_ouput_dir + '/{}'.format(st_idx))
            # eval
            # apd_mean += eval_util.compute_apd(test_out_lst)
            # ade, fde = eval_util.compute_ade(test_out_lst, ref_clip_0)
            # ade_mean += ade
            # fde_mean += fde

        #     test_data_long = test_data_long.detach().cpu().numpy()
        #     test_out_long_lst = []
        #     for i in range(test_data_long.shape[0]):
        #         ###
        #         # tmp_test_long = copy.deepcopy(test_data_long[i]).reshape(copy.deepcopy(test_data_long[i]).shape[0] * self.block_size,self.frame_dim)
        #         # denorm_tmp_test_long = self.dataset.denorm_data(tmp_test_long).reshape(int(tmp_test_long.shape[0] / self.block_size),self.frame_dim * self.block_size)
        #         denorm_tmp_test_long = self.dataset.denorm_data(copy.deepcopy(test_data_long[i]))
        #
        #         cur_denormed_test_data = denorm_tmp_test_long
        #         cur_jnts = []
        #
        #         for mode in self.dataset.data_component:
        #             jnts_mode = self.dataset.x_to_jnts(cur_denormed_test_data, mode=mode)
        #             cur_jnts.append(jnts_mode)
        #
        #             if mode == self.dataset.data_component[0]:
        #                 test_out_long_lst.append(jnts_mode)
        #                 jnts_mode_local = jnts_mode - jnts_mode[:, [0], :]
        #         cur_jnts = np.array(cur_jnts)
        #
        #     test_out_long_lst = np.array(test_out_long_lst)
        #     self.plot_traj_fn(test_out_long_lst, result_ouput_dir + '/{}_long'.format(st_idx))
        # print("apd", apd_mean)
        # print("ade", ade_mean)
        # print("fde", fde_mean)

        return NaN_clip_num