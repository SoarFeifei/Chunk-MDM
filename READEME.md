# <p align="center"> Chunk-MDM: Adaptive Temporal Chunked Motion Diffusion Model for Long-Horizon Character Control </p>

## Dataset Preparation
For each dataset, our dataloader automatically parses it into a sequence of 1D frames, saving the frames as data.npz and the essential normalization statistics as stats.npz within your dataset directory. We provide stats.npz so users can perform inference without needing to download the full dataset and provide a single file from the dataset instead.

### LaFAN1:
[Download](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and extract under ```./data/LAFAN``` directory.
BEWARE: We didn't include files with a prefix of 'obstacle' in our experiments. 

### 100STYLE:
[Download](https://www.ianxmason.com/100style/) and extract under ```./data/100STYLE``` directory.

### Arbitrary BVH dataset:
Download and extract under ```./data/``` directory. Create a yaml config file in ```./config/model/```, 

### AMASS:
Follow the procedure described in the repo of [HuMoR](https://github.com/davrempe/humor) and extract under ```./data/AMASS``` directory.

## Installation
```
conda create -n amdm python=3.7
conda activate cmdm
pip install -r requirement.txt
mkdir output
```

## Base Model
### Training
```
python run_base.py --arg_file args/amdm_DATASET_train.txt
```
### Inference
```
python run_env.py --arg_file args/RP_amdm_DATASET.txt
```
### TargetReaching
```
python run_env.py --arg_file args/TG_amdm_DATASET.txt
```

## Acknowledgement
We deeply appreciate the authors of **[A-MDM]** for their foundational codebase, which serves as the core model for this repository. Their contributions have been invaluable to the development of this project.

If you find this project helpful, please also consider supporting and citing the original A-MDM work:
- **Repository:** [(https://github.com/Yi-Shi94/AMDM)]
- **Paper:** [(https://arxiv.org/pdf/2306.00416v4)]
