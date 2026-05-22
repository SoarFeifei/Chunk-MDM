import glob
import numpy as np

import dataset.base_dataset as base_dataset
import dataset.util.amass as amass_util
import dataset.util.plot as plot_util
import os.path as osp

class AMASS(base_dataset.BaseMotionData):
    NAME = 'AMASS'
    def __init__(self, config):
        super().__init__(config)
        
    def plot_jnts(self, x, path=None):
        return plot_util.plot_jnt_vel(x, self.links, plot_util.plot_lafan1, self.fps, path)
        
    def plot_traj(self, x, path=None):
        return plot_util.plot_traj_lafan1(x, path)
    
    def get_motion_fpaths(self):
        path =  osp.join(self.path,'**/*.{}'.format('npz'))
        file_lst = glob.glob(path, recursive = True)
        return file_lst
    def get_motion_fpaths_text(self):
        babel_json_path = self.config["data"]["babel_path"]
        with open(babel_json_path, 'r') as f:
            babel_data = json.load(f)
            
        segment_lst = []
        
        # 遍历 BABEL 的每一个条目
        for sid, seq_data in babel_data.items():
            feat_p = seq_data.get("feat_p", None)
            if feat_p is None:
                continue
            fps = feat_p.rsplit('_', 2)
            if fps[1:3] != '30':
                print("!!! not 30fps")
            parts = feat_p.rsplit('_', 4)  # 分割最后4个下划线部分
            result = parts[0]
                
            # 拼接本地 AMASS 的绝对路径
            amass_file_path = osp.join(self.path, result)
            if not osp.exists(amass_file_path):
                print("!!! not exit ",amass_file_path)
                continue
                
            # 提取帧级标注 (frame_ann)
            if "frame_ann" in seq_data and seq_data["frame_ann"] is not None:
                for label_info in seq_data["frame_ann"]["labels"]:
                    proc_label = label_info.get("proc_label", "")
                    start_t = label_info.get("start_t", 0.0)
                    end_t = label_info.get("end_t", 0.0)
                    
                    # 过滤掉 transition (过渡动作通常语义不明确)
                    # if proc_label == "transition" or (end_t - start_t) <= 0:
                    #     continue
                        
                    segment_lst.append({
                        "path": amass_file_path,
                        "text": proc_label,
                        "start_t": start_t,
                        "end_t": end_t
                    })
        return segment_lst # 返回切片任务列表，而不是纯路径
    
    def process_data(self, fname):
        motion_struct = amass_util.init_motion_from_amass(fname)
        offset_feature = motion_struct._skeleton.get_joint_offset()
        offset_feature = np.array(offset_feature).reshape(1,-1)
        
        xs = amass_util.load_amass_file(fname)
        offset_feature = offset_feature.repeat(xs.shape[0],0)
        xs = np.concatenate([xs, offset_feature],axis=-1)
        return xs, motion_struct

    ###
    def load_new_data(self, path):
        motion, motion_struct = self.process_data(path)
        x_normed = self.norm_data(motion)
        ###
        x_normed = x_normed.reshape(-1)
        index = int(x_normed.shape[0] - (x_normed.shape[0] % self.frame_dim))
        x_normed = x_normed[:index]
        x_normed = x_normed.reshape(-1, self.frame_dim)
        return x_normed
    
    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        idx_ = self.valid_idx[idx]
        motion = self.motion_flattened[idx_:idx_+self.rollout]
        return  motion


if __name__=='__main__':
    pass
