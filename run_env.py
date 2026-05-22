import warnings
warnings.filterwarnings("ignore")

import os
os.environ['WANDB_API_KEY'] = '...'
os.environ['WANDB_ENTITY'] = '...'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import shutil
import torch
import numpy as np

import dataset.dataset_builder as dataset_builder
import model.model_builder as model_builder
import model.trainer_builder as trainer_builder
import policy.envs.env_builder as env_builder
import policy.learning.agent_builder as agent_builder

from policy.common.misc_utils import EpisodeRunner
import util.arg_parser as arg_parser
import util.rand_util as rand_util
import util.mp_util as mp_util

import time
import os.path as osp
import random
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, print("Failed to load args from: " + arg_file)

    rand_seed_key = "rand_seed"
    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
        #rand_seed = mp_util.get_proc_rank()
        rand_util.set_rand_seed(rand_seed)
    return args


def build_trainer(config, device):
    trainer = trainer_builder.build_trainer(config, device)
    return trainer

def build_model(config, dataset, device):
    model = model_builder.build_model(config, dataset, device)
    return model

def build_dataset(config, load_full_dataset):
    dataset = dataset_builder.build_dataset(config, load_full_dataset)
    return dataset

def build_agent(config, model, env, device):
    agent = agent_builder.build_agent(config, model, env, device)
    return agent

def build_env(config, int_output_dir, model, dataset, mode, device):
    env = env_builder.build_envs(config, int_output_dir, model, dataset, mode, device)
    return env

def train(agent, out_model_file, int_output_dir):
    agent.train_controller(out_model_file=out_model_file, 
                      int_output_dir=int_output_dir)
    return


def evaluate(agent):
    agent.evaluate_controller()
    return 


def test(agent):
    agent.test_controller()
    return 

from dataset.util.skeleton_info import LAFAN1_links
from dataset.util.skeleton_info import SMPL_links
import util.eval as eval_util
import copy
def test_no_agent(env, valid_range):

    ###evaluate
    if env.dataset.conds_flag:
        conds = {}
        traj_pose = torch.zeros(1000, env.block_size, 1).unsqueeze(0).to(env.device)  # .unsqueeze(0).to(env.device)
        traj_trans = torch.zeros(1000, env.block_size, 2).unsqueeze(0).to(env.device)
        conds["traj_pose"] = traj_pose.reshape(1000, -1)
        conds["traj_trans"] = traj_trans.reshape(1000, -1)
        ref_clip = env.dataset.motion_flattened[0:int(120 / env.dataset.block_size) + 1]
        start_x = torch.from_numpy(ref_clip[0, :]).float().to(env.device)
        start_x = start_x.reshape(1, -1)
        # start = time.time()
        # test_data_long = env.model.eval_seq(start_x, conds, 1000, 1)    # b, n, f
        # end = time.time()
        # print(f"Block-AMDM generate {1000} frames in {end - start:.3f}s")
        # print(f"FPS = {1000 / (end - start):.2f}")
        # print(f"Speedup = {(1000 / (end - start)) / (1000 / (end - start)):.2f}x")

        ### to joint
        ref_clip = ref_clip.reshape(ref_clip.shape[0] * env.dataset.block_size, int(env.dataset.frame_dim / env.dataset.block_size))
        # print("ref_clip", ref_clip.shape)   # ref_clip (60, 267)
        # denorm_ref_clip = self.dataset.denorm_data(ref_clip).reshape(int(ref_clip.shape[0] / self.block_size),self.frame_dim * self.block_size)
        denorm_ref_clip = env.dataset.denorm_data(ref_clip)
        # print("denorm_ref_clip", denorm_ref_clip.shape)
        ref_clip = env.dataset.x_to_jnts(denorm_ref_clip, mode=env.dataset.data_component[0])[None, ...]
        sk_length = eval_util.extract_sk_lengths(LAFAN1_links, ref_clip)
        sk_length = sk_length[:, 0]
        # std_per_bone = np.std(sk_length, axis=1)
        # print("sk_length", sk_length.mean(), sk_length.max(), sk_length.min(), sk_length.shape)
        # print("std_per_bone", std_per_bone)

        # # print("test_data_long", test_data_long.shape)
        # pred_long = test_data_long[0, :, :].detach().cpu().numpy()
        # denorm_pred_long = env.dataset.denorm_data(copy.deepcopy(pred_long))
        # # print("denorm_ref_clip", denorm_ref_clip.shape)
        # pred_long = env.dataset.x_to_jnts(denorm_pred_long, mode=env.dataset.data_component[0])[None, ...]
        # sk_length = eval_util.extract_sk_lengths(LAFAN1_links, pred_long)
        # print("sk_length_pred", sk_length.mean(), sk_length.max(), sk_length.min())

    else:
        plot_traj_fn = env.dataset.plot_traj if hasattr(env.dataset, 'plot_traj') and callable(env.dataset.plot_traj) \
            else vis_util.vis_traj
        plot_jnts_fn = env.dataset.plot_jnts if hasattr(env.dataset, 'plot_jnts') and callable(
            env.dataset.plot_jnts) \
                else vis_util.vis_skel

        # normed_motion = env.dataset.load_new_data("../AMDM_origin/AMDM/data/AMASS/BioMotionLab_NTroje/rub036/0004_motorcycle_poses_93_frames_30_fps.npz")
        # normed_motion = normed_motion.reshape(-1, 333)
        # print("normed_motion", normed_motion.shape)
        # cur_denormed_test_data = env.dataset.denorm_data(copy.deepcopy(normed_motion))
        # cur_jnts = []
        # for mode in env.dataset.data_component:
        #     jnts_mode = env.dataset.x_to_jnts(cur_denormed_test_data, mode=mode)
        #     cur_jnts.append(jnts_mode)
        #
        #     if mode == env.dataset.data_component[0]:
        #         ### save npy
        #         np.save('./output/base/amdm_amass/' + '_AMASS_348', jnts_mode.astype(np.float32))
        # print("save 11111")
        #
        # st_idx = 100
        # conds = None
        # test_num = 50
        # motion_length = 60
        # ###
        # if env.dataset.dataset_name == "LAFAN1":
        #     # start_i = [0, 7344, 11288, 16205, 23551, 28292, 35625, 39557]
        #     valid_range = np.array(valid_range)
        #     print("valid_range", valid_range)
        #     start_i = []
        #     for i_f, (idx_st, idx_ed) in enumerate(valid_range):
        #         start_i.append(idx_st)
        #     result_ouput_dir = './output/base/amdm_lafan1'
        # elif env.dataset.dataset_name == "STYLE100":
        #     result_ouput_dir = './output/base/amdm_100style'
        #     valid_range = np.array(valid_range)
        #     print("valid_range", valid_range)
        #     start_i = []
        #     for i_f, (idx_st, idx_ed) in enumerate(valid_range):
        #         start_i.append(idx_st)
        # elif env.dataset.dataset_name == "AMASS":
        #     result_ouput_dir = './output/base/amdm_amass'
        #     valid_range = np.array(valid_range)
        #     print("valid_range", valid_range)
        #     start_i = []
        #     for i_f, (idx_st, idx_ed) in enumerate(valid_range):
        #         start_i.append(idx_st)
        # foot_slide = bone_err = dist_mean = dist_min = dist_min_last = joint_acc = 0
        # pen_freq = 0
        # pen_dist = 0
        # for i in range(len(start_i)):   # len(start_i)
        #     st_idx = i * 100
        #     # ref_clip = env.dataset.motion_flattened[0:int(120 / env.dataset.block_size) + 1]
        #     ref_clip = env.dataset.motion_flattened[start_i[i] + 50:start_i[i] + int(motion_length/env.dataset.block_size) + 50 ]
        #     # print("env.dataset.motion_flattened", env.dataset.motion_flattened.shape)
        #     start_x = torch.from_numpy(ref_clip[0, :]).float().to(env.device)
        #     start_x = start_x.reshape(1, -1)
        #     test_data_long = env.model.eval_seq(start_x, conds, motion_length, test_num)
        #
        #     # start = time.time()
        #     # test_data_long = env.model.eval_seq(start_x, conds, 60, 1)
        #     # end = time.time()
        #     # print(f"Block-AMDM generate {60} frames in {end - start:.3f}s")
        #     # print(f"FPS = {1000 / (end - start):.2f}")
        #     # print(f"Speedup = {(1000 / (end - start)) / (1000 / (end - start)):.2f}x")
        #
        #     ref_clip = ref_clip.reshape(ref_clip.shape[0] * env.dataset.block_size,
        #                                 int(env.dataset.frame_dim / env.dataset.block_size))
        #     denorm_ref_clip = env.dataset.denorm_data(ref_clip)
        #     ref_clip = env.dataset.x_to_jnts(denorm_ref_clip, mode=env.dataset.data_component[0])[None, ...]
        #     cur_jnts = []
        #     for mode in env.dataset.data_component:
        #         jnts_mode = env.dataset.x_to_jnts(denorm_ref_clip, mode=mode)
        #         cur_jnts.append(jnts_mode)
        #     cur_jnts = np.array(cur_jnts)
        #     plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/{}_ref'.format(st_idx))   ### gif
        #     if env.dataset.dataset_name == "LAFAN1" or env.dataset.dataset_name == "STYLE100":
        #         links = LAFAN1_links
        #     elif env.dataset.dataset_name == "AMASS":
        #         links = SMPL_links
        #     sk_length = eval_util.extract_sk_lengths(links, ref_clip)
        #     sk_length = sk_length[:, :1]
        #
        #     ### gif
        #     test_data_long = test_data_long.detach().cpu().numpy()
        #     test_out_long_lst = []
        #     for j in range(test_data_long.shape[0]):
        #         cur_denormed_test_data = env.dataset.denorm_data(copy.deepcopy(test_data_long[j]))
        #         cur_jnts = []
        #
        #         for mode in env.dataset.data_component:
        #             jnts_mode = env.dataset.x_to_jnts(cur_denormed_test_data, mode=mode)
        #             cur_jnts.append(jnts_mode)
        #
        #             if mode == env.dataset.data_component[0]:
        #                 test_out_long_lst.append(jnts_mode)
        #                 ### save npy
        #                 # if j == 0:
        #                 #     np.save(result_ouput_dir + '/{}_SMPL_180'.format(st_idx), jnts_mode.astype(np.float32))
        #
        #         pred_long = test_data_long[j, :, :]  # .detach().cpu().numpy()
        #         denorm_pred_long = env.dataset.denorm_data(copy.deepcopy(pred_long))
        #         pred_long_jnts = env.dataset.x_to_jnts(denorm_pred_long, mode=env.dataset.data_component[0])[
        #             None, ...]
        #         pred_long_jnts = pred_long_jnts.squeeze(0)
        #         # print("pred_long_jnts", pred_long_jnts.shape)     # (150, 22, 3)
        #         sk_length_pred = eval_util.extract_sk_lengths(links, pred_long_jnts)
        #         # print("sk_length_pred", sk_length_pred.mean(), sk_length_pred.max(), sk_length_pred.min())
        #
        #         bone_err_per_frame = np.abs(sk_length_pred - sk_length)  # [21,125]
        #         # 平均 Bone Error
        #         bone_err += bone_err_per_frame.mean() / test_num
        #
        #         # 穿地
        #         if env.dataset.dataset_name == "LAFAN1":
        #             foot_idx = [3, 4, 7, 8]
        #         elif env.dataset.dataset_name == "STYLE100":
        #             foot_idx = [17, 18, 21, 22]
        #         elif env.dataset.dataset_name == "AMASS":
        #             foot_idx = [7, 8, 10, 11]
        #         contact_zs_mean, contact_event = eval_util.compute_ground_pen(foot_idx, pred_long_jnts, -0.03)
        #         # print("contact_zs_mean contact_event", contact_zs_mean, contact_event)
        #         pen_freq += contact_event / test_num
        #         pen_dist += contact_zs_mean / test_num
        #     #             jnts_mode_local = jnts_mode - jnts_mode[:, [0], :]
        #     #     cur_jnts = np.array(cur_jnts)
        #     #     plot_jnts_fn(cur_jnts.squeeze(), result_ouput_dir + '/{}_{}'.format(st_idx, i))
        #     test_out_long_lst = np.array(test_out_long_lst)
        #     plot_traj_fn(test_out_long_lst, result_ouput_dir + '/{}_long'.format(st_idx))
        #
        #     # print("test_data_long", test_data_long.shape)     # (1, 150, 267)
        #
        #     print("pen_freq pen_dist", pen_freq * 5, pen_dist * 5)
        #
        #     # 脚滑
        #     foot_slide += eval_util.compute_foot_slide(foot_idx, test_out_long_lst)
        #     print("foot_slide", foot_slide * 5)
        #
        #     print("bone_err", bone_err * 5)
        #     # apd
        #     dist_mean += eval_util.compute_apd(test_out_long_lst)
        #     print("dist_mean", dist_mean * 5)
        #     # ade
        #     dist_min_i, dist_min_last_i, min_idx = eval_util.compute_ade(test_out_long_lst, ref_clip)
        #     dist_min += dist_min_i
        #     dist_min_last += dist_min_last_i
        #     print("dist_min, dist_min_last", dist_min * 5, dist_min_last * 5)
        #     # joint Acc
        #     joint_acc += eval_util.compute_Acc(test_out_long_lst)
        #     print("joint_acc", joint_acc * 5)

        #     ### save npy
        #     # if st_idx == 400 :#or st_idx == 600 :
        #     #     print("st_idx:2 - 6")
        #     #     denormed_min_data = env.dataset.denorm_data(copy.deepcopy(test_data_long[min_idx]))
        #     #     min_jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=mode)
        #     #     np.save(result_ouput_dir + '/{}_SMPL_180'.format(st_idx), min_jnts_mode.astype(np.float32))
        #     #
        #     #     min_cur_jnts = []
        #     #     for mode in env.dataset.data_component:
        #     #         jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=mode)
        #     #         min_cur_jnts.append(jnts_mode)
        #     #     min_cur_jnts = np.array(min_cur_jnts)
        #     #     plot_jnts_fn(min_cur_jnts.squeeze(), result_ouput_dir + '/{}_pred_{}'.format(st_idx, min_idx))  ### gif
        #     #
        #     #     min_idx -= 4
        #     #     denormed_min_data = env.dataset.denorm_data(copy.deepcopy(test_data_long[min_idx]))
        #     #     min_jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=mode)
        #     #     np.save(result_ouput_dir + '/{}_SMPL_180_{}'.format(st_idx, min_idx), min_jnts_mode.astype(np.float32))
        #     #
        #     #     min_cur_jnts = []
        #     #     for mode in env.dataset.data_component:
        #     #         jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=mode)
        #     #         min_cur_jnts.append(jnts_mode)
        #     #     min_cur_jnts = np.array(min_cur_jnts)
        #     #     plot_jnts_fn(min_cur_jnts.squeeze(), result_ouput_dir + '/{}_pred_{}'.format(st_idx, min_idx))  ### gif

    env.reset()
    env.reset_initial_frames()
    ###
    traj = 'circle'
    max_step = 0.06  # 单帧最大速度
    T_frame = 28
    if traj == 'circle':
        radius = 6.0  # 圆半径
        omega = 0.6  # 角速度（rad / step）

        # 圆心（可以是固定点，也可以随 env 变化）
        # center = torch.zeros_like(env.root_xz)  # (B,2)，以原点为圆心
        center = env.root_xz.clone()
        center[:, 0] += radius

        phi0 = torch.pi  # 起点在左侧
        phis = phi0 + omega * torch.arange(T_frame, device=env.device)

        x = center[:, 0] + radius * torch.cos(phis)
        z = center[:, 1] + radius * torch.sin(phis)

        positions = torch.stack([x, z], dim=-1)
        print("positions", positions.shape)
    elif traj == 's':
        amp = 1.0
        period = 40
        t = torch.arange(T_frame, device=env.device).float()
        # 向前（沿 -z）
        z = -0.4 * t
        # 左右摆动（x 方向）
        x = amp * torch.sin(2 * torch.pi * t / period)
        positions = torch.stack([x, z], dim=-1)
        # 平移到起点
        positions = positions + env.root_xz
    elif traj == 'forward':
        t = torch.arange(T_frame, device=env.device).float()
        dx = torch.zeros_like(t)
        dz = -max_step * t * 10
        positions = torch.stack([dx, dz], dim=-1)
        positions = positions + env.root_xz
    # positions = [[-3, -3], [-2, 4], [8, 0], [10, 2], [8, 0],[6,6]]
    positions = [[0, -5], [-3, -7], [-6, -5], [-6, 0], [4,0], [7,0]]
    positions = torch.tensor(positions)
    with EpisodeRunner(env) as runner:

        ### conditions
        traj_pose_buffer = []  # 每个元素: (block_size,)
        traj_trans_buffer = []  # 每个元素: (block_size, 2)

        # target_root = torch.rand(2).unsqueeze(0).to(env.device) * 20 -
        target_index = 0
        target_root = torch.tensor(positions[target_index]).unsqueeze(0).to(env.device)
        traj_pose = torch.zeros(env.block_size, 1).unsqueeze(0).to(env.device)  # .unsqueeze(0).to(env.device)
        traj_trans = torch.zeros(env.block_size, 2).unsqueeze(0).to(env.device)

        if env.dataset.conds_flag:
            conds = {}
            env.target_markers(target_root)

        step_t = 0
        while not runner.done:
        # for step in range(300):
            start_x = env.get_cond_frame().to(env.device)
            ### conditions
            if env.dataset.conds_flag:
                # max_step = 0.10
                raw_displacement = target_root - env.root_xz
                displacement_norm = torch.norm(raw_displacement)

                dx = target_root[:, 0] - env.root_xz[:, 0]
                dz = target_root[:, 1] - env.root_xz[:, 1]
                target_angle = torch.atan2(dx, -dz)  # shape: [B]
                target_angle = (target_angle + 2 * torch.pi) % (2 * torch.pi)
                cur_angle = env.root_facing.squeeze(-1)  # shape [B]
                angle_diff = (target_angle - cur_angle + torch.pi) % (2 * torch.pi) - torch.pi
                delta_theta = torch.clamp(angle_diff, -0.04, 0.04)
                traj_pose[:, :, :] = delta_theta

                next_facing = env.root_facing# + delta_theta.unsqueeze(-1)
                next_facing = next_facing.remainder_(2 * np.pi)
                # ===== 4. 计算速度：沿 next_facing 方向前进 =====
                direction = torch.cat([
                    torch.sin(next_facing),
                    -torch.cos(next_facing)
                    # torch.cos(next_facing),
                    # torch.sin(next_facing)
                ], dim=-1)  # (B,2)
                # 单帧最大速度
                root_vel_global = direction * max_step  # 世界坐标速度
                # ===== 5. 转到 canonical 坐标系 (no-heading) =====
                # 构造逆旋转矩阵 R^-1 = R(−facing)
                cos_f = torch.cos(-env.root_facing)
                sin_f = torch.sin(-env.root_facing)
                # batch 版本旋转矩阵
                R_inv = torch.stack([
                    torch.stack([cos_f, -sin_f], dim=-1),
                    torch.stack([sin_f, cos_f], dim=-1)
                ], dim=-2).squeeze(0)  # (B,2,2)
                # (B,2,2) × (B,2,1) → (B,2)
                root_vel_local = torch.matmul(R_inv, root_vel_global.unsqueeze(-1)).squeeze(-1)
                traj_trans[:, :, ] = root_vel_local

                conds["traj_pose"] = traj_pose.reshape(1, -1)
                conds["traj_trans"] = traj_trans.reshape(1, -1)

                ###
                traj_pose_buffer.append(
                    traj_pose.clone().reshape(-1)  # (block_size)
                )
                # print(traj_pose_buffer)

                traj_trans_buffer.append(
                    traj_trans.clone().reshape(-1)  # (block_size * 2)
                )

                frame = env.get_next_frame(conds=conds)

            else:
                step_t += 1
                frame = env.get_next_frame()
                # print("frame", frame.shape)
            # if step_t == 152:
            #     start_x = env.get_cond_frame().to(env.device)
            #     pred_round_t = env.model.eval_seq(start_x, None, 240, 1)  # b, n, f
            #     pred_round = pred_round_t.detach().cpu().numpy()
            #     denormed_min_data = env.dataset.denorm_data(copy.deepcopy(pred_round[0]))
            #     min_jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=env.dataset.data_component[0])
            #     result_ouput_dir = './output/base/amdm_lafan1'
            #     np.save(result_ouput_dir + '/random_SMPL_240', min_jnts_mode.astype(np.float32))

            for i in range(env.frame_skip):
                _, reward, done, info = env.calc_env_state(frame)
                # print("reward",reward)
                if done.any():
                    reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                    env.reset_index(reset_indices)
                #try:
                #    if info.get("reset").all():
                #        env.reset()
                #except:
                #    if info.get("reset"):
                #        env.reset()
            if env.dataset.conds_flag:
                if displacement_norm < 0.4:
                    # target_root += torch.rand(2).unsqueeze(0).to(env.device) * 20 - 10
                    target_index += 1
                    target_root = torch.tensor(positions[target_index]).unsqueeze(0).to(env.device)
                    env.target_markers(target_root)
                    if target_index == 5:
                        traj_pose_all = torch.stack(traj_pose_buffer, dim=0)   # shape: (n, block_size)
                        traj_trans_all = torch.stack(traj_trans_buffer, dim=0)  # shape: (n, block_size* 2)\
                        print("traj_pose_buffer", traj_pose_all.shape, traj_trans_all.shape)

                        conds["traj_pose"] = traj_pose_all  # [48:, :]
                        conds["traj_trans"] = traj_trans_all  # [48:, :]
                        print("conds", traj_pose_all.shape)
                        start_x = start_x.reshape(1, -1)
                        pred_num = 1
                        pred_round_t = env.model.eval_seq(start_x, conds, 520, pred_num)    # b, n, f   # 405 635
                        pred_round = pred_round_t.detach().cpu().numpy()
                        print("pred_round_t", pred_round_t.shape)

                        ### traj
                        root_xz = torch.zeros(pred_num, 2)
                        root_facing = torch.zeros(pred_num, 1)
                        traj_history = []
                        colors = cm.jet(np.linspace(0, 1, pred_num))
                        for block_j in range(int(520 / 5)):
                            pose = pred_round_t[:, block_j * 5:(block_j+1)*5, :].cpu().view(pred_num, -1)
                            for i in range(env.block_size):
                                # print("pose", pose.shape)
                                single_dim = int(env.dataset.frame_dim / env.block_size)
                                pose_t = pose[:, i * single_dim:(i + 1) * single_dim]
                                pose_denorm = env.dataset.denorm_data(pose_t)
                                dr = env.dataset.get_heading_dr(pose_denorm)[..., None]
                                root_xz_vel = env.dataset.get_root_linear_planar_vel(pose_denorm)
                                root_facing.add_(dr).remainder_(2 * np.pi)
                                root_rotmat_up = env.get_rotation_matrix(root_facing)
                                displacement = (root_rotmat_up * root_xz_vel.unsqueeze(1)).sum(dim=2)
                                root_xz.add_(displacement)
                                # print("self.root_xz", self.root_xz[0])
                                current_xz = root_xz.cpu().numpy()
                                traj_history.append(current_xz.copy())
                        # ... 原有计算 reward 和 done 的逻辑 ...
                        # 【触发可视化】当某个环境完成寻路（或者 max_timestep 结束）

                        traj = np.array(traj_history)
                        plt.figure(figsize=(5, 5))
                        # 画出角色的实际行走轨迹
                        for b in range(pred_num):
                            # 画出角色的实际行走轨迹
                            plt.plot(traj[:, b, 0], traj[:, b, 1], color=colors[b], alpha=0.7)
                            # 标记起点 (方块)
                            plt.scatter(traj[0, b, 0], traj[0, b, 1], color=colors[b], marker='s')
                            # 标记终点 (叉号，方便你看清哪边是头哪边是尾，不需要可注释掉)
                            # plt.scatter(traj[-1, b, 0], traj[-1, b, 1], color=colors[b], marker='x')
                        if torch.is_tensor(positions):
                            pts = positions[:4, :].detach().cpu().numpy()
                        else:
                            pts = np.array(positions[:4, :])


                        # 使用 scatter 绘制   个点
                        # s=150: 显著增大标记尺寸
                        # marker='*': 使用五角星标记，醒目
                        # color='red': 使用纯红色，区别于轨迹颜色
                        # zorder=10: 确保这16个点画在轨迹线的上方，不被遮挡
                        plt.scatter(pts[:, 0], pts[:, 1],
                                    s=150,
                                    marker='*',
                                    color='black',
                                    label='Specific Points',
                                    edgecolors='black',  # 给五角星加个黑边，更清晰
                                    zorder=10)
                        # 标记目标点
                        plt.xlabel("World X")
                        plt.ylabel("World Z")
                        plt.axis('equal')
                        plt.grid(True)
                        plt.legend()
                        # 保存到本地，不要阻塞训练
                        result_ouput_dir = './output/base/amdm_lafan1/target2.png'
                        plt.savefig(result_ouput_dir)
                        plt.close()

                        ### traj
                        # test_data_long = pred_round
                        # test_out_long_lst = []
                        # for j in range(test_data_long.shape[0]):
                        #     cur_denormed_test_data = env.dataset.denorm_data(copy.deepcopy(test_data_long[j]))
                        #     cur_jnts = []
                        #
                        #     for mode in env.dataset.data_component:
                        #         jnts_mode = env.dataset.x_to_jnts(cur_denormed_test_data, mode=mode)
                        #         cur_jnts.append(jnts_mode)
                        #
                        #         if mode == env.dataset.data_component[0]:
                        #             test_out_long_lst.append(jnts_mode)
                        # test_out_long_lst = np.array(test_out_long_lst)
                        # plot_traj_fn = env.dataset.plot_traj if hasattr(env.dataset, 'plot_traj') and callable(
                        #     env.dataset.plot_traj) \
                        #     else vis_util.vis_traj
                        # result_ouput_dir = './output/base/amdm_lafan1'
                        # plot_traj_fn(test_out_long_lst, result_ouput_dir + '/circle_long')

                        ### save npy
                        # # print("pred_round", pred_round.shape)
                        denormed_min_data = env.dataset.denorm_data(copy.deepcopy(pred_round[0]))
                        min_jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=env.dataset.data_component[0])
                        result_ouput_dir = './output/base/amdm_lafan1'
                        np.save(result_ouput_dir + '/target2_SMPL_200', min_jnts_mode.astype(np.float32))
    return      


def create_output_dirs(out_model_file, int_output_dir):
    if (mp_util.is_root_proc()):
        output_dir = os.path.dirname(out_model_file)
        if (output_dir != "" and (not os.path.exists(output_dir))):
            os.makedirs(output_dir, exist_ok=True)
        
        if (int_output_dir != "" and (not os.path.exists(int_output_dir))):
            os.makedirs(int_output_dir, exist_ok=True)
    return

def copy_config_file(config_file, output_dir):
    out_file = os.path.join(output_dir, os.path.basename(config_file))
    shutil.copy(config_file, out_file)
    return


            
def run(rank, num_procs, args):
    mode = args.parse_string("mode", "train")
    device = args.parse_string("device", 'cuda:0')
    
    test_motion_file = args.parse_string("test_motion_file", "")
    test_motion_frame = args.parse_string("test_motion_frame", "")

    out_model_file = args.parse_string("out_model_file", "")
    trained_model_path = args.parse_string("model_path", "")
    int_output_dir = args.parse_string("int_output_dir", "")
    master_port = args.parse_string("master_port", "")
    env_config_file = args.parse_string("env_config", "")
    model_config_file = args.parse_string("model_config", "")
    agent_config_file = args.parse_string("agent_config", "")
    trained_controller_path = args.parse_string("controller_path", "")
    mp_util.init(rank, num_procs, device, master_port)

    set_np_formatting()
    #if out_model_file is not None and int_output_dir is not None:
    create_output_dirs(out_model_file, int_output_dir)
    out_model_dir = os.path.dirname(out_model_file)
    
    load_full_motion = mode == 'train' or test_motion_file == ""
    dataset = build_dataset(model_config_file, load_full_motion)
    if test_motion_file != "":
        print('Loading test file:', test_motion_file)
        normed_motion = dataset.load_new_data(test_motion_file) # (1468, 1335)
        
        if test_motion_frame != "":
            test_motion_frame = int(test_motion_frame)
            normed_motion = normed_motion[test_motion_frame,:].reshape(-1, normed_motion.shape[-1])
            
        dataset.motion_flattened = normed_motion
        dataset.valid_range = [0,dataset.motion_flattened.shape[0]]
        dataset.valid_idx = np.arange(0,dataset.motion_flattened.shape[0])

        valid_range = list()
        ###
        start_i = [0, 1468, 2256, 3239, 4708, 5656, 7122, 7908]
        generage_length = 2    ### 60 / 3
        if dataset.dataset_name == "LAFAN1":
            bvh_files = []
            for root, dirs, files in os.walk('./data/LAFAN1'):
                for file in files:
                    if file.endswith('.bvh'):
                        full_path = os.path.join(root, file)
                        bvh_files.append(full_path)

            random.seed(11451)
            selected_files = random.sample(bvh_files, min(2, len(bvh_files)))
            # print(selected_files)
            total_len = 0
            for i, file_path in enumerate(selected_files):
                normed_motion = dataset.load_new_data(file_path)
                dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)
                length = normed_motion.shape[0]
                valid_range.append([total_len, total_len + length])
                total_len += length
            # normed_motion = dataset.load_new_data("data/LAFAN1/dance1_subject2.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (2256, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/fallAndGetUp2_subject3.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (3239, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/fight1_subject2.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (4708, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/ground1_subject4.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (5656, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/jumps1_subject2.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (7122, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/obstacles3_subject3.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (7908, 1335)
            # normed_motion = dataset.load_new_data("data/LAFAN1/walk1_subject5.bvh")
            # dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)    # (9475, 1335)
        elif dataset.dataset_name == "STYLE100":
            bvh_files = []

            for root, dirs, files in os.walk('../AMDM_origin/AMDM/data/100STYLE'):
                for file in files:
                    if file.endswith('.bvh'):
                        full_path = os.path.join(root, file)
                        bvh_files.append(full_path)

            random.seed(11451)
            selected_files = random.sample(bvh_files, min(20, len(bvh_files)))
            # print(selected_files)
            total_len = 0
            for i, file_path in enumerate(selected_files):
                normed_motion = dataset.load_new_data(file_path)
                dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)
                length = normed_motion.shape[0]
                valid_range.append([total_len, total_len + length])
                total_len += length
        elif dataset.dataset_name == "AMASS":
            path = osp.join('../AMDM_origin/AMDM/data/AMASS', '**/*.{}'.format('npz'))
            file_lst = glob.glob(path, recursive=True)
            random.seed(11451)
            selected_files = random.sample(file_lst, min(40, len(file_lst)))
            total_len = 0
            for i, file_path in enumerate(selected_files):
                normed_motion = dataset.load_new_data(file_path)
                length = normed_motion.shape[0]
                if length >= generage_length:
                    dataset.motion_flattened = np.concatenate([dataset.motion_flattened, normed_motion], axis=0)
                    valid_range.append([total_len, total_len + length])
                    total_len += length
                    print("select ", file_path)

                if len(valid_range) >= 20:
                    break

    else:
        if test_motion_frame != "":
            test_motion_frame = int(test_motion_frame)
            dataset.motion_flattened = dataset.motion_flattened[test_motion_frame].reshape(-1, dataset.motion_flattened.shape[-1])
            dataset.valid_range = [0,dataset.motion_flattened.shape[0]]
            dataset.valid_idx = np.arange(0,dataset.motion_flattened.shape[0])

    if trained_model_path:
        try:
            print('Loading model param:{}\n model config:{}'.format(trained_model_path, model_config_file))
            model = model_builder.build_model(model_config_file, dataset, device)
            state_dict = torch.load(trained_model_path)

            model.load_state_dict(state_dict)

        except:
            print('Loading model: {}'.format(trained_model_path))
            model = torch.load(trained_model_path)
        
        model.to(device)
        model.eval()
    else:
        model = None

    if agent_config_file:
        env = build_env(env_config_file, int_output_dir, model, dataset, mode, device)
        agent = build_agent(agent_config_file, model, env, device)
        if trained_controller_path:
            print("Loading controller:",trained_controller_path)
            
            try:
                actor_critic = agent.actor_critic
                state_dict = torch.load(trained_controller_path)
                actor_critic.load_state_dict(state_dict)
            except:
                actor_critic = torch.load(trained_controller_path)
        
            actor_critic.to(device)
            actor_critic.eval()
            agent.actor_critic = actor_critic
    else:   
        env = build_env(env_config_file, int_output_dir, model, dataset, 'test', device)
        agent = None
    
    if (mode == "train"):
        assert agent is not None, "require a controller & a agent"
        copy_config_file(agent_config_file, out_model_dir)
        copy_config_file(env_config_file, out_model_dir)
        copy_config_file(model_config_file, out_model_dir)
        train(agent, out_model_file=out_model_file, int_output_dir=int_output_dir)
   
    elif (mode == "test"):
        if agent is None:

            # normed_motion_1 = dataset.load_new_data(selected_files[0])
            # normed_motion_1 = normed_motion_1.reshape(-1, 267)
            # denormed_min_data = env.dataset.denorm_data(normed_motion_1[:180])
            # min_jnts_mode = env.dataset.x_to_jnts(denormed_min_data, mode=env.dataset.data_component[0])
            # print("jnts", min_jnts_mode.shape)
            # result_ouput_dir = './output/base/amdm_lafan1'
            # np.save(result_ouput_dir + '/bvh1_SMPL_180', min_jnts_mode.astype(np.float32))

            print('agent is None, test no agent')
            test_no_agent(env, valid_range) ###
        else:

            test(agent)

    elif (mode == "eval"):
        evaluate(agent)

    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    args = load_args(argv)
    num_workers = args.parse_int("num_workers", 1)
    assert(num_workers > 0)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for i in range(num_workers - 1):
        rank = i + 1
        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, args])
        proc.start()
        processes.append(proc)

    run(0, num_workers, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)
