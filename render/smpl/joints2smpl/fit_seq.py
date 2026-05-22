from __future__ import print_function, division
import argparse
import torch
import os,sys
from os import walk, listdir
from os.path import isfile, join
import numpy as np
import joblib
import smplx
import trimesh
import h5py
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from smplify import SMPLify3D
import config

###
LAFAN1_name_joint = [
    'Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe',
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
]

# --- 映射规则定义 ---
mapping = {
    # Hips -> MidHip
    'Hips': 'MidHip',
    # 手部置换: Hand -> Wrist
    'LeftHand': 'LWrist',
    'RightHand': 'RWrist',
    # Spine/Neck/Head/Arms/Legs: 直接映射
    'LeftUpLeg': 'LHip', 'LeftLeg': 'LKnee', 'LeftFoot': 'LAnkle',
    'RightUpLeg': 'RHip', 'RightLeg': 'RKnee', 'RightFoot': 'RAnkle',
    'Spine': 'spine1', 'Spine1': 'spine2', 'Spine2': 'spine3',
    'Neck': 'Neck', 'Head': 'Head',
    'LeftShoulder': 'LShoulder', 'LeftArm': 'LElbow', 'LeftForeArm': 'LShoulder', # 假设 ForeArm -> Elbow (取决于 SMPL 链)
    'RightShoulder': 'RShoulder', 'RightArm': 'RElbow', 'RightForeArm': 'RShoulder', # 假设 ForeArm -> Elbow (取决于 SMPL 链)
    # 脚趾置换: Toe -> Foot
    'LeftToe': 'LFoot',
    'RightToe': 'RFoot'
}

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=100,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=False,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=22,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS",
                    help='use correspondence')
parser.add_argument('--fix_foot', type=str, default="False",
                    help='fix foot or not')
parser.add_argument('--data_folder', type=str, default="./demo/demo_data",
                    help='data in the folder')
parser.add_argument('--save_folder', type=str, default="./demo/demo_results",
                    help='results save folder')
parser.add_argument('--files', type=str, default="test_motion.npy",
                    help='files use')
opt = parser.parse_args()
print(opt)

# ---load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(config.SMPL_MODEL_DIR)
smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                         model_type="smpl", gender="neutral", ext="pkl",
                         batch_size=opt.batchSize).to(device)

# ## --- load the mean pose as original ---- 
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, 'r')
init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
#
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

# # #-------------initialize SMPLify
smplify = SMPLify3D(smplxmodel=smplmodel,
                    batch_size=opt.batchSize,
                    joints_category=opt.joint_category,
					num_iters=opt.num_smplify_iters,
                    device=device)
#print("initialize SMPLify3D done!")

    
# purename_list = os.path.splitext(opt.files)
purename_list = ['dart_joints_r_s','dart_joints']
# print("purename_list", purename_list)
# --- load data ---
def fit_seq(purename):
	print("purename", purename)
	data = np.load(opt.data_folder + "/" + purename + ".npy")  # [nframes, njoints, 3]
	data = data.reshape(-1, 22,3)
	print("data", data.shape)

	dir_save = os.path.join(opt.save_folder, purename)
	if not os.path.isdir(dir_save):
		os.makedirs(dir_save, exist_ok=True)

	# run the whole seqs
	num_seqs = data.shape[0]

	###
	data = data[:, :, [2, 0, 1]]
	data[:, :, 2] = -data[:, :, 2]
	data[:, :, 1] = -data[:, :, 1]
	### 初始化目标 AMASS 格式坐标数组 (N 帧, 22 关节, 3D)
	# J_amass_coords = np.zeros((num_seqs, 22, 3)) # 注意：AMASS 22 关节，不含 Toes End
	# for lafan_name, amass_name in mapping.items():
	# 	if lafan_name in LAFAN1_name_joint and amass_name in config.AMASS_JOINT_MAP:
	# 		lafan_idx = LAFAN1_name_joint.index(lafan_name)
	# 		amass_idx = config.AMASS_JOINT_MAP[amass_name]
	#
	# 		# 将 LAFAN1 的坐标数据复制到 AMASS 对应的索引位置
	# 		J_amass_coords[:, amass_idx, :] = data[:, lafan_idx, :]
	# data = J_amass_coords

	for idx in tqdm(range(num_seqs)):
		#print(idx)

		joints3d = data[idx] #*1.2 #scale problem [check first]
		keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

		if idx == 0:
			pred_betas[0, :] = init_mean_shape
			pred_pose[0, :] = init_mean_pose
			pred_cam_t[0, :] = cam_trans_zero
		else:
			data_param = joblib.load(dir_save + "/" + "%04d"%(idx-1) + ".pkl")
			pred_betas[0, :] = torch.from_numpy(data_param['beta']).unsqueeze(0).float()
			pred_pose[0, :] = torch.from_numpy(data_param['pose']).unsqueeze(0).float()
			pred_cam_t[0, :] = torch.from_numpy(data_param['cam']).unsqueeze(0).float()

		if opt.joint_category =="AMASS":
			confidence_input =  torch.ones(opt.num_joints)
			# make sure the foot and ankle
			if opt.fix_foot == True:
				confidence_input[7] = 1.5
				confidence_input[8] = 1.5
				confidence_input[10] = 1.5
				confidence_input[11] = 1.5
			###
			# confidence_input[13] = 0.01
			# confidence_input[14] = 0.01
		else:
			if opt.fix_foot == True:
				confidence_input[7] = 1.5
				confidence_input[8] = 1.5
				confidence_input[10] = 1.5
				confidence_input[11] = 1.5
			print("Such category not settle down!")

		# ----- from initial to fitting -------
		new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
		new_opt_cam_t, new_opt_joint_loss = smplify(
													pred_pose.detach(),
													pred_betas.detach(),
													pred_cam_t.detach(),
													keypoints_3d,
													conf_3d=confidence_input.to(device),
													seq_ind=idx
													)
		# print("new_opt_cam_t", new_opt_cam_t.shape)
		# # -- save the results to ply---
		outputp = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
							transl=new_opt_cam_t.view(-1, 3), return_verts=True)	###
		mesh_p = trimesh.Trimesh(vertices=outputp.vertices.detach().cpu().numpy().squeeze(), faces=smplmodel.faces, process=False)
		mesh_p.export(dir_save + "/" + "%04d"%idx + ".ply")

		# save the pkl
		param = {}
		param['beta'] = new_opt_betas.detach().cpu().numpy()
		param['pose'] = new_opt_pose.detach().cpu().numpy()
		param['cam'] = new_opt_cam_t.detach().cpu().numpy()
		joblib.dump(param, dir_save + "/" + "%04d"%idx + ".pkl", compress=3)

for i in range(len(purename_list)):
	fit_seq(purename_list[i])
