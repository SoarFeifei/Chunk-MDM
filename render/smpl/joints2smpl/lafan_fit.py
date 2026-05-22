import argparse
import os

import joblib
import numpy as np
import smplx
import torch
import trimesh
from tqdm import tqdm

from src.smplify import SMPLify3D


LAFAN1_NAMES = [
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

AMASS_JOINT_MAP = {
    "MidHip": 0, "LHip": 1, "RHip": 2, "spine1": 3, "LKnee": 4, "RKnee": 5,
    "spine2": 6, "LAnkle": 7, "RAnkle": 8, "spine3": 9, "LFoot": 10, "RFoot": 11,
    "Neck": 12, "LCollar": 13, "Rcollar": 14, "Head": 15, "LShoulder": 16,
    "RShoulder": 17, "LElbow": 18, "RElbow": 19, "LWrist": 20, "RWrist": 21,
}

DIRECT_MAP = {
    "Hips": "MidHip",
    "LeftUpLeg": "LHip",
    "RightUpLeg": "RHip",
    "Spine": "spine1",
    "LeftLeg": "LKnee",
    "RightLeg": "RKnee",
    "Spine1": "spine2",
    "LeftFoot": "LAnkle",
    "RightFoot": "RAnkle",
    "Spine2": "spine3",
    "LeftToe": "LFoot",
    "RightToe": "RFoot",
    "Neck": "Neck",
    "Head": "Head",
    "LeftShoulder": "LCollar",
    "RightShoulder": "Rcollar",
    "LeftArm": "LShoulder",
    "RightArm": "RShoulder",
    "LeftForeArm": "LElbow",
    "RightForeArm": "RElbow",
    "LeftHand": "LWrist",
    "RightHand": "RWrist",
}


def get_lafan1_to_amass_indices():
    src_indices = []
    tgt_indices = []
    for src_name, tgt_name in DIRECT_MAP.items():
        src_indices.append(LAFAN1_NAMES.index(src_name))
        tgt_indices.append(AMASS_JOINT_MAP[tgt_name])
    return np.asarray(src_indices, dtype=np.int64), np.asarray(tgt_indices, dtype=np.int64)


def load_and_convert_lafan1(data_path):
    lafan1_data = np.load(data_path).astype(np.float32, copy=False)
    lafan1_data = lafan1_data[:, :, [0, 2, 1]]
    lafan1_data[:, :, 2] *= -1.0
    lafan1_data[:, :, 1] *= -1.0
    lafan1_data[:, :, 0] *= -1.0
    return lafan1_data


def build_target_joints(lafan1_data, device):
    num_frames = lafan1_data.shape[0]
    src_indices, tgt_indices = get_lafan1_to_amass_indices()
    lafan1_tensor = torch.from_numpy(lafan1_data).to(device=device, dtype=torch.float32)
    target_joints = torch.zeros((num_frames, 22, 3), device=device, dtype=torch.float32)
    target_joints[:, tgt_indices] = lafan1_tensor[:, src_indices]
    return target_joints


def fit_lafan1_sequence(
    data_path,
    save_dir,
    model_path,
    num_smplify_iters=100,
    save_mesh=True,
    save_pkl=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lafan1_data = load_and_convert_lafan1(data_path)
    target_joints_all = build_target_joints(lafan1_data, device)
    num_frames = target_joints_all.shape[0]

    smpl_model = smplx.create(
        model_path,
        model_type="smpl",
        gender="neutral",
        batch_size=1,
    ).to(device)
    smplify = SMPLify3D(
        smplxmodel=smpl_model,
        batch_size=1,
        joints_category="AMASS",
        num_iters=num_smplify_iters,
        device=device,
    )

    confidence = torch.ones(22, device=device, dtype=torch.float32)
    curr_pose = torch.zeros(1, 72, device=device, dtype=torch.float32)
    curr_betas = torch.zeros(1, 10, device=device, dtype=torch.float32)
    curr_cam_t = torch.zeros(1, 3, device=device, dtype=torch.float32)

    os.makedirs(save_dir, exist_ok=True)
    sequence_name = os.path.splitext(os.path.basename(data_path))[0]
    output_dir = os.path.join(save_dir, sequence_name)
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(num_frames), desc="Fitting Sequence"):
        frame_target_joints = target_joints_all[idx:idx + 1]
        _, _, new_opt_pose, new_opt_betas, new_opt_cam_t, _ = smplify(
            curr_pose.detach(),
            curr_betas.detach(),
            curr_cam_t.detach(),
            frame_target_joints,
            conf_3d=confidence,
            seq_ind=idx,
        )

        curr_pose = new_opt_pose
        curr_betas = new_opt_betas
        curr_cam_t = new_opt_cam_t

        frame_prefix = os.path.join(output_dir, f"{idx:04d}")

        if save_mesh:
            with torch.no_grad():
                output = smpl_model(
                    betas=new_opt_betas,
                    global_orient=new_opt_pose[:, :3],
                    body_pose=new_opt_pose[:, 3:],
                    transl=new_opt_cam_t.view(-1, 3),
                    return_verts=True,
                )
            mesh = trimesh.Trimesh(
                vertices=output.vertices.detach().cpu().numpy().squeeze(0),
                faces=smpl_model.faces,
                process=False,
            )
            mesh.export(f"{frame_prefix}.ply")

        if save_pkl:
            param = {
                "beta": new_opt_betas.detach().cpu().numpy(),
                "pose": new_opt_pose.detach().cpu().numpy(),
                "cam": new_opt_cam_t.detach().cpu().numpy(),
            }
            joblib.dump(param, f"{frame_prefix}.pkl", compress=3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./demo/Lafan_data/target12_SMPL_200.npy")
    parser.add_argument("--save_dir", type=str, default="./demo/Lafan_results/")
    parser.add_argument("--model_path", type=str, default="./body_models/")
    parser.add_argument("--num_smplify_iters", type=int, default=100)
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_pkl", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fit_lafan1_sequence(
        data_path=args.data_path,
        save_dir=args.save_dir,
        model_path=args.model_path,
        num_smplify_iters=args.num_smplify_iters,
        save_mesh=not args.skip_mesh,
        save_pkl=not args.skip_pkl,
    )
