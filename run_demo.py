import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import dataset.dataset_builder as dataset_builder
import model.model_builder as model_builder
from dataset.util.humanml3d.util.metrics import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from util.arg_parser import ArgParser
import util.vis_util as vis_util


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FID on BABEL val_amass_matched.json using AMASS motion features. "
            "Frame-level labels are aligned to motion blocks, and variable-length sequences "
            "are evaluated with length buckets."
        )
    )
    parser.add_argument("--arg_file", type=str, default="args/RP_amdm_amass.txt",
                        help="Optional run_env-style arg file.")
    parser.add_argument(
        "--babel_json",
        type=str,
        default="data/babel/val_amass_matched.json",
        help="BABEL validation json with matched feat_p paths.",
    )
    parser.add_argument(
        "--amass_root",
        type=str,
        default="data/AMASS",
        help="Root directory used to resolve relative feat_p paths.",
    )
    parser.add_argument("--model_config", type=str, default="", help="Model config yaml.")
    parser.add_argument("--model_path", type=str, default="", help="Trained model checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Evaluate at most this many samples. -1 means all.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        help="Skip clips shorter than this frame count.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Skip clips longer than this frame count.",
    )
    parser.add_argument(
        "--bucket_batch_size",
        type=int,
        default=16,
        help="Batch size inside each same-length bucket.",
    )
    parser.add_argument(
        "--save_details",
        type=str,
        default="",
        help="Optional json path to save per-sample metadata.",
    )
    return parser.parse_args()


def build_dataset(model_config_file: str):
    return dataset_builder.build_dataset(model_config_file, load_full_dataset=False)


def load_model(model_config_file: str, model_path: str, dataset, device: str):
    try:
        print(f"Loading model param:{model_path}\n model config:{model_config_file}")
        model = model_builder.build_model(model_config_file, dataset, device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception:
        print(f"Loading serialized model object: {model_path}")
        model = torch.load(model_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def merge_with_arg_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.arg_file:
        if not args.model_config or not args.model_path:
            raise ValueError("Either provide --model_config and --model_path, or pass --arg_file.")
        return args

    arg_parser = ArgParser()
    ok = arg_parser.load_file(args.arg_file)
    if not ok:
        raise ValueError(f"Failed to load arg file: {args.arg_file}")

    if not args.model_config:
        args.model_config = arg_parser.parse_string("model_config", "")
    if not args.model_path:
        args.model_path = arg_parser.parse_string("model_path", "")
    if args.device == "cuda:0":
        args.device = arg_parser.parse_string("device", args.device)

    if not args.model_config or not args.model_path:
        raise ValueError("model_config/model_path are still empty after reading --arg_file.")
    return args


def load_babel_json(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


class TextConditionCache:
    def __init__(self, dataset, device: str):
        self.dataset = dataset
        self.device = device
        self.cache: Dict[str, Optional[torch.Tensor]] = {}

    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        if text in self.cache:
            return self.cache[text]

        if not text:
            self.cache[text] = None
            return None

        cond = None
        if hasattr(self.dataset, "get_clip_class_embedding"):
            cond = self.dataset.get_clip_class_embedding([text], outformat="pt")
        elif hasattr(self.dataset, "text_encoder"):
            _, cond = self.dataset.text_encoder([text])
            cond = self._to_2d_float_tensor(cond)

        if cond is not None:
            cond = cond.to(self.device)
        self.cache[text] = cond
        return cond

    def _to_2d_float_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).float()
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("text_encoder returned an empty sequence.")
            if len(value) == 1 and isinstance(value[0], (torch.Tensor, np.ndarray)):
                return self._to_2d_float_tensor(value[0])

            items = [self._to_2d_float_tensor(item) for item in value]
            if all(item.shape == items[0].shape for item in items):
                tensor = torch.cat(items, dim=0)
            else:
                raise ValueError(
                    f"text_encoder returned inconsistent shapes: {[tuple(item.shape) for item in items]}"
                )
        else:
            tensor = torch.tensor(value, dtype=torch.float32)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def build_batch_extra_info(self, block_texts: List[str]) -> Optional[torch.Tensor]:
        conds: List[torch.Tensor] = []
        for text in block_texts:
            cond = self.encode_text(text)
            if cond is None:
                return None
            conds.append(cond)

        if not conds:
            return None
        return torch.cat(conds, dim=0)


def resolve_feat_path(feat_path: str, amass_root: str) -> str:
    if os.path.isabs(feat_path):
        return feat_path
    if feat_path.startswith("data/"):
        return feat_path
    return os.path.join(amass_root, feat_path)


def pred_seq(model, start_x, text, num_steps, text_cache: TextConditionCache, device):
    cur_x = torch.from_numpy(start_x).to(device).unsqueeze(0).to(dtype=torch.long)
    print("cur_x", cur_x.shape)
    outputs = torch.zeros((num_steps, start_x.shape[-1]),
                          dtype=cur_x.dtype,
                          device=device,
                          )
    extra_info = text_cache.build_batch_extra_info(text)
    for step_idx in range(num_steps):
        cur_x = model.eval_step(cur_x, extra_info).detach()
        outputs[step_idx, :] = cur_x
    return outputs
def pred_seq_no_cond(model, start_x, num_steps, device):
    cur_x = torch.from_numpy(start_x).to(device).unsqueeze(0).to(dtype=torch.long)
    print("cur_x", cur_x.shape)
    block_size = 5
    outputs = torch.zeros((num_steps * block_size, int(start_x.shape[-1]/block_size)),
                          dtype=cur_x.dtype,
                          device=device,
                          )
    for step_idx in range(num_steps):
        cur_x = cur_x.reshape(1, -1)
        cur_x = model.eval_step(cur_x).detach()
        cur_x = cur_x.reshape(block_size, -1)
        for i in range(block_size):
            outputs[step_idx * block_size + i, :] = cur_x[i, :]
    return outputs


def main() -> None:
    args = merge_with_arg_file(parse_cli())
    dataset = build_dataset(args.model_config)
    model = load_model(args.model_config, args.model_path, dataset, args.device)
    # text_cache = TextConditionCache(dataset, args.device)
    # babel_data = load_babel_json(args.babel_json)
    #
    # items = list(babel_data.items())
    # if args.max_samples > 0:
    #     items = items[: args.max_samples]

    # for idx, (sample_id, sample) in enumerate(items, start=1):
    # feat_path = sample.get("feat_p")
    # feat_path = 'KIT/348/bend_left01_poses_95_frames_30_fps.npz'
    # amass_root = 'data/babel/val_amass_matched.json'

    dataset_root = 'data/LAFAN1/'
    feat_path = 'multipleActions1_subject2'   # 'jumps1_subject2'[3480:3900, :] 'fallAndGetUp1_subject4'[740:900, :]
    # dataset_root = '../AMDM_origin/AMDM/data/AMASS'
    # root_path = 'Transitions_mocap/mazen_c3d/'
    # feat_path = 'JOOF_kick_poses_183_frames_30_fps'
    motion_path = resolve_feat_path(feat_path + '.bvh', dataset_root)
    # motion_path = resolve_feat_path(root_path + feat_path + '.npz', dataset_root)
    motion = dataset.load_new_data(motion_path)
    block_size = dataset.block_size
    print("11")

    plot_jnts_fn = dataset.plot_jnts if hasattr(dataset, 'plot_jnts') and callable(
        dataset.plot_jnts) \
        else vis_util.vis_skel
    ref_clip = motion.reshape(-1, int(dataset.frame_dim/dataset.block_size))
    denorm_x = dataset.denorm_data(ref_clip[555:1400, :])
    # denorm_x = dataset.denorm_data(ref_clip[:9, :])
    cur_jnts = []
    for mode in dataset.data_component:
        jnts_mode = dataset.x_to_jnts(denorm_x, mode=mode)
        cur_jnts.append(jnts_mode)
    # np.save('output/demo/' + feat_path + '_gt_', jnts_mode.astype(np.float32))
    cur_jnts = np.array(cur_jnts)

    plot_jnts_fn(cur_jnts.squeeze(), 'output/demo/' + feat_path + '_gt')
    # print("22")

    # text = ['stand']
    # start_x = motion[1000]
    # pred_x = pred_seq_no_cond(model, start_x, 19, args.device)
    # pred_x = pred_x.detach().cpu().numpy()
    # print("pred_x", pred_x.shape)
    # denorm_x = dataset.denorm_data(pred_x)
    # cur_jnts = []
    # for mode in dataset.data_component:
    #     jnts_mode = dataset.x_to_jnts(denorm_x, mode=mode)
    #     cur_jnts.append(jnts_mode)
    # cur_jnts = np.array(cur_jnts)
    # plot_jnts_fn(cur_jnts.squeeze(), 'output/demo' + '/text_pred')

    start_x = torch.from_numpy(motion[int(555/block_size)]).to(args.device).to(dtype=torch.long)

    # past_seq = torch.from_numpy(motion[:2, :]).to(args.device).to(dtype=torch.long)
    # past_seq = past_seq.expand(10, 2, 999)
    # past_seq = past_seq.reshape(10, -1, int(dataset.frame_dim / dataset.block_size))

    pred_seq = model.eval_seq(start_x, None, 180, 10)

    # pred_seq = torch.cat((past_seq, pred_seq), dim=1)
    pred_seq = pred_seq.detach().cpu().numpy()

    print("pred_x", pred_seq.shape)
    if pred_seq.shape[0] > 1:
        for i in range(pred_seq.shape[0]):
            denorm_seq = dataset.denorm_data(pred_seq[i,:,:])
            cur_jnts = []
            for mode in dataset.data_component:
                jnts_mode = dataset.x_to_jnts(denorm_seq, mode=mode)
                cur_jnts.append(jnts_mode)
                # if i == 1:
            np.save('output/demo/' + feat_path + '_pred_{}'.format(i), jnts_mode.astype(np.float32))
            cur_jnts = np.array(cur_jnts)
            # cur_jnts.reshape(1,-1,22,3)
            # print("cur_jnts", cur_jnts.shape)
            plot_jnts_fn(cur_jnts.squeeze(), 'output/demo/' + feat_path + '_pred_{}'.format(i))
    else:
        pred_seq = pred_seq.squeeze(0)
        denorm_seq = dataset.denorm_data(pred_seq)
        cur_jnts = []
        for mode in dataset.data_component:
            jnts_mode = dataset.x_to_jnts(denorm_seq, mode=mode)
            cur_jnts.append(jnts_mode)
        cur_jnts = np.array(cur_jnts)
        plot_jnts_fn(cur_jnts.squeeze(), 'output/demo/' + feat_path + '_pred')

    # start_x = torch.from_numpy(np.stack([s["start_x"] for s in bucket_samples], axis=0)).to(device)


if __name__ == "__main__":
    main()