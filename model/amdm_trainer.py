import numpy as np

import torch
from tqdm import tqdm
import model.trainer_base as trainer_base


class AMDMTrainer(trainer_base.BaseTrainer):
    NAME = 'AMDM'

    def __init__(self, config, dataset, device):
        super(AMDMTrainer, self).__init__(config, dataset, device)
        optimizer_config = config['optimizer']
        self.full_T = optimizer_config.get('full_T', False)
        self.consistency_on = optimizer_config.get('consistency_on', False)
        self.consist_loss_weight = optimizer_config.get('consist_loss_weight', 1)
        self.loss_type = config["diffusion"]["loss_type"]
        self.recon_on = optimizer_config.get('recon_on', False)
        self.recon_loss_weight = optimizer_config.get('recon_loss_weight', 1)
        self.diffusion_loss_weight = optimizer_config.get('diffusion_loss_weight', 1)
        self.detach_step = optimizer_config.get('detach_step', 3)

        self.cond_loss_weight = 0.1

    def compute_rpr_consist_loss(self, last_frame, cur_frame):
        cur_frame_denormed = self.dataset.denorm_data(cur_frame, device=cur_frame.device)
        last_frame_denormed = self.dataset.denorm_data(last_frame, device=last_frame.device)

        jnts = self.dataset.jnts_frame_pt(cur_frame_denormed)
        if self.loss_type == 'l1':
            loss_fn = torch.nn.functional.l1_loss
        else:
            loss_fn = torch.nn.functional.mse_loss

        if 'angle' in self.dataset.data_component:
            jnts_fk = self.dataset.angle_frame_pt(cur_frame_denormed)
            consist_loss_fk_jnts = loss_fn(jnts_fk, jnts.squeeze())
        else:
            consist_loss_fk_jnts = 0

        if 'velocity' in self.dataset.data_component:
            jnts_vel = self.dataset.vel_frame_pt(last_frame_denormed, cur_frame_denormed)
            consist_loss_vel_jnts = loss_fn(jnts_vel, jnts.squeeze())
        else:
            consist_loss_vel_jnts = 0
        # consist_loss_fk_vel = loss_fn(jnts_vel, jnts_fk)
        return consist_loss_fk_jnts + consist_loss_vel_jnts

    def compute_teacher_loss(self, model, sampled_frames, extra_info):
        # st_index = random.randint(0,sampled_frames.shape[1]-2)
        # print('teacher forcing')
        last_frame = sampled_frames[:, 0, :]
        ground_truth = sampled_frames[:, 1, :]
        self.optimizer.zero_grad()

        diff_loss, pred_frame, total_err, loss_traj = model.compute_loss(last_frame, ground_truth, None, extra_info)  ### conditions
        loss = self.diffusion_loss_weight * diff_loss + loss_traj * self.cond_loss_weight  # + 0.000025 * total_err   # smooth_loss targetloss

        loss.backward()
        self.optimizer.step()
        model.update()

        smooth_err = total_err  #

        return {"diff_loss": diff_loss.item()}, loss_traj

    def compute_student_loss(self, model, sampled_frames, sch_samp_prob, extra_info):
        # print('student forcing')
        loss_diff_sum, loss_consist_sum = 0, 0
        # overlap
        smooth_err = 0

        batch_size = sampled_frames.shape[0]
        shrinked_batch_size = batch_size // model.T

        for st_index in range(self.num_rollout - 1):    # 3-1
            self.optimizer.zero_grad()
            next_index = st_index + 1
            ground_truth = sampled_frames[:, next_index, :]

            if self.full_T:
                shrinked_batch_size = batch_size
                if st_index == 0:
                    last_frame = sampled_frames[:, 0, :]
                    last_frame_expanded = last_frame[:, None, :].expand(-1, model.T, -1).reshape(
                        shrinked_batch_size * model.T, -1)
                else:
                    last_frame = pred_frame.detach().reshape(shrinked_batch_size, model.T, -1)[:, 0, :]
                    teacher_forcing_mask = torch.bernoulli(
                        1.0 - torch.ones(shrinked_batch_size, device=pred_frame.device) * sch_samp_prob).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
                    last_frame_expanded = last_frame[:, None, :].expand(-1, model.T, -1).reshape(
                        shrinked_batch_size * model.T, -1)

                ground_truth_expanded = ground_truth[:, None, :].expand(-1, model.T, -1).reshape(
                    shrinked_batch_size * model.T, -1)

                ts = torch.arange(0, model.T, device=self.device)
                ts = ts[None, ...].expand(shrinked_batch_size, -1).reshape(-1)

                diff_loss, pred_frame, total_err, loss_traj = model.compute_loss(last_frame_expanded, ground_truth_expanded, ts, extra_info)
                loss = self.diffusion_loss_weight * diff_loss + loss_traj * self.cond_loss_weight  # + 0.000025 * total_err   # smooth_loss targetloss


            else:
                if st_index == 0:
                    last_frame = sampled_frames[:, 0, :]
                else:
                    last_frame = pred_frame.detach()
                    teacher_forcing_mask = torch.bernoulli(
                        1.0 - torch.ones(batch_size, device=pred_frame.device) * sch_samp_prob).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
                    # last_frame_expanded = last_frame[:,None,:].expand(-1, model.T, -1).reshape(shrinked_batch_size*model.T, -1)
                    # ts = torch.zeros(batch_size, device=self.device).long()

                diff_loss_teacher, _, total_err, loss_traj = model.compute_loss(sampled_frames[:, st_index, :], ground_truth, None,
                                                          extra_info)
                diff_loss_student, pred_frame, total_err, loss_traj = model.compute_loss(last_frame, ground_truth, None, extra_info)
                diff_loss = diff_loss_student + diff_loss_teacher  # / self.num_rollout
                loss = self.diffusion_loss_weight * diff_loss + loss_traj * self.cond_loss_weight  # + 0.000025 * total_err   # smooth_loss

            loss.backward()
            self.optimizer.step()
            model.update()

            loss_diff_sum += diff_loss.item()

            smooth_err += total_err     #

        return {"diff_loss": loss_diff_sum}, loss_traj

    def train_loop(self, ep, model):
        ep_loss_dict = {}

        num_samples = 0
        self._update_lr_schedule(self.optimizer, ep - 1)

        model.train()
        pbar = tqdm(self.train_dataloader, colour='green')
        cur_samples = 1
        # print("pbar", len(pbar))    # 20
        # print("self.num_rollout ", self.num_rollout)    # 3
        error = 0

        ### conditions
        if self.dataset.conds_flag:
            for datas in pbar:
                # print("frames",frames.shape)      #torch.Size([4096, 3, 1335])
                # extra_info = None
                datas = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in datas.items()}
                cond = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in
                        datas['conditions'].items()}
                frames = datas['data'].to(self.device).float()
                extra_info = cond       ### cond
                # print("frames.shape",frames.shape)
                # print("self.sample_schedule[ep].shape",self.sample_schedule[ep].shape)

                self.optimizer.zero_grad()

                if self.sample_schedule[ep] > 0:
                    loss_dict, loss_traj = self.compute_student_loss(model, frames, self.sample_schedule[ep], extra_info=extra_info)
                else:
                    loss_dict, loss_traj = self.compute_teacher_loss(model, frames, extra_info=extra_info)

                num_samples += cur_samples

                loss = 0
                for key in loss_dict:
                    # print("key", key)   # diff_loss
                    loss += loss_dict[key]
                    if key not in ep_loss_dict:
                        ep_loss_dict[key] = loss_dict[key]
                    else:
                        ep_loss_dict[key] += loss_dict[key]

                out_str = ' '.join(['{}:{:.4f}'.format(key, val) for key, val in loss_dict.items()])
                pbar.set_description('ep:{}, {}'.format(ep, out_str))
        else:
            for frames in pbar:
                # print("frames",frames.shape)      #torch.Size([4096, 3, 1335])
                extra_info = None

                frames = frames.to(self.device).float()
                # print("frames.shape",frames.shape)
                # print("self.sample_schedule[ep].shape",self.sample_schedule[ep].shape)

                self.optimizer.zero_grad()

                if self.sample_schedule[ep] > 0:
                    loss_dict, loss_traj = self.compute_student_loss(model, frames, self.sample_schedule[ep],
                                                                       extra_info=extra_info)
                else:
                    loss_dict, loss_traj = self.compute_teacher_loss(model, frames, extra_info=extra_info)

                num_samples += cur_samples

                loss = 0
                for key in loss_dict:
                    # print("key", key)   # diff_loss
                    loss += loss_dict[key]
                    if key not in ep_loss_dict:
                        ep_loss_dict[key] = loss_dict[key]
                    else:
                        ep_loss_dict[key] += loss_dict[key]

                out_str = ' '.join(['{}:{:.4f}'.format(key, val) for key, val in loss_dict.items()])
                pbar.set_description('ep:{}, {}'.format(ep, out_str))

        for key in loss_dict:
            ep_loss_dict[key] /= num_samples

        train_info = {
            "epoch": ep,
            "sch_smp_rate": self.sample_schedule[ep],
            **ep_loss_dict,
            # "smooth_err": error / num_samples,
            "loss_target": loss_traj
        }

        return train_info
