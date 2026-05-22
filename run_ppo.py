import warnings
warnings.filterwarnings("ignore")

import os
os.environ['WANDB_API_KEY'] = '...'
os.environ['WANDB_ENTITY'] = '...'

import sys
import shutil
import torch
import numpy as np
import yaml

import dataset.dataset_builder as dataset_builder
import model.model_builder as model_builder
import policy.envs.env_builder as env_builder
import util.arg_parser as arg_parser
import util.rand_util as rand_util
import util.mp_util as mp_util

from policy.learning.ppo_agent_with_physics import PPOAgentWithPhysics
from policy.learning.ppo_model_physics import PPOModel


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, print("Failed to load args from: " + arg_file)

    rand_seed_key = "rand_seed"
    if args.has_key(rand_seed_key):
        rand_seed = args.parse_string(rand_seed_key)
        rand_seed = int(rand_seed)
        print('rand seed', rand_seed)
        rand_util.set_rand_seed(rand_seed)
    return args


def load_yaml_file(file):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def create_output_dirs(out_model_file, int_output_dir):
    if mp_util.is_root_proc():
        output_dir = os.path.dirname(out_model_file)
        if output_dir != "" and (not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        if int_output_dir != "" and (not os.path.exists(int_output_dir)):
            os.makedirs(int_output_dir, exist_ok=True)


def copy_config_file(config_file, output_dir):
    out_file = os.path.join(output_dir, "config.yaml")
    shutil.copy(config_file, out_file)


def build_dataset(config_file):
    config = load_yaml_file(config_file)
    dataset = dataset_builder.build_dataset(config_file, load_full_data=True)
    return dataset, config


def build_bmdm_model(model_config_file, dataset, device, trained_model_path=None):
    config = load_yaml_file(model_config_file)
    model = model_builder.build_model(model_config_file, dataset, device)

    if trained_model_path and os.path.exists(trained_model_path):
        try:
            print('Loading BMDM model param: {}'.format(trained_model_path))
            state_dict = torch.load(trained_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print('BMDM model loaded successfully')
        except Exception as e:
            print('Loading BMDM model directly: {}'.format(trained_model_path))
            model = torch.load(trained_model_path, map_location=device)

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, config


def build_ppo_config(agent_config_file, env_config, model_config):
    agent_config = load_yaml_file(agent_config_file)

    agent_config['block_size'] = env_config.get('block_size', 5)
    agent_config['use_physics_constraints'] = agent_config.get('use_physics_constraints', True)
    agent_config['use_bmdm_integration'] = agent_config.get('use_bmdm_integration', True)
    agent_config['physics_weight'] = agent_config.get('physics_weight', 1.0)
    agent_config['bmdm_guide_weight'] = agent_config.get('bmdm_guide_weight', 0.5)

    agent_config['physics_constraints'] = agent_config.get('physics_constraints', {
        'foot_slide_weight': 1.0,
        'floating_weight': 1.0,
        'ground_penetration_weight': 1.0,
        'smoothness_weight': 0.1,
        'contact_threshold': 0.05,
        'slide_threshold': 0.02,
        'ground_threshold': 0.0
    })

    agent_config['bmdm_explorer'] = agent_config.get('bmdm_explorer', {
        'num_exploration_samples': 5,
        'exploration_noise_scale': 0.1,
        'use_best_of_n': True
    })

    agent_config['bmdm_guide'] = agent_config.get('bmdm_guide', {
        'guide_weight': 0.5,
        'similarity_threshold': 0.8
    })

    return agent_config


def build_env(env_config_file, int_output_dir, model, dataset, mode, device):
    env = env_builder.build_envs(env_config_file, int_output_dir, model, dataset, mode, device)
    return env


def build_ppo_agent(config, env, device, bmdm_model=None):
    actor_critic = PPOModel(config=config, bmdm_model=bmdm_model, env=env, device=device)

    agent = PPOAgentWithPhysics(
        config=config,
        actor_critic=actor_critic,
        env=env,
        device=device,
        bmdm_model=bmdm_model
    )

    return agent


def run(rank, num_procs, args):
    mode = args.parse_string("mode", "train")
    device = args.parse_string("device", "cuda:0")
    log_file = args.parse_string("log_file", "")

    out_model_file = args.parse_string("out_model_file", "output/rl/bmdm_rl.pth")
    int_output_dir = args.parse_string("int_output_dir", "output/rl/")

    bmdm_model_path = args.parse_string("bmdm_model_path", "output/base/amdm_lafan1/_ep2280_B5.pth")
    model_config_file = args.parse_string("model_config", "config/model/amdm_lafan1.yaml")
    env_config_file = args.parse_string("env_config", "config/envs/randomplay.yaml")
    agent_config_file = args.parse_string("agent_config", "config/agents/BMDM_ppo.yaml")

    mp_util.init(rank, num_procs, device, "0")

    set_np_formatting()
    create_output_dirs(out_model_file, int_output_dir)

    print("=" * 60)
    print("Building Dataset")
    print("=" * 60)
    dataset, model_config = build_dataset(model_config_file)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Frame dim: {dataset.frame_dim}")

    print("\n" + "=" * 60)
    print("Building Pre-trained BMDM Model")
    print("=" * 60)
    bmdm_model, _ = build_bmdm_model(
        model_config_file,
        dataset,
        device,
        trained_model_path=bmdm_model_path if bmdm_model_path else None
    )
    print(f"BMDM model built, parameters frozen: {not any(p.requires_grad for p in bmdm_model.parameters())}")

    print("\n" + "=" * 60)
    print("Building Environment")
    print("=" * 60)
    env = build_env(env_config_file, int_output_dir, bmdm_model, dataset, mode, device)
    print(f"Environment: {env.NAME}")
    print(f"Action dim: {env.action_dim}")
    print(f"Observation space: {env.observation_space.shape}")

    print("\n" + "=" * 60)
    print("Building PPO Agent with Physics Constraints")
    print("=" * 60)
    env_config = load_yaml_file(env_config_file)
    ppo_config = build_ppo_config(agent_config_file, env_config, model_config)
    agent = build_ppo_agent(ppo_config, env, device, bmdm_model)
    print(f"PPO Agent built: {agent.NAME}")
    print(f"Physics constraints enabled: {agent.use_physics_constraints}")
    print(f"BMDM integration enabled: {agent.use_bmdm_integration}")

    if mode == "train":
        copy_config_file(agent_config_file, os.path.dirname(out_model_file))
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        agent.train_controller(out_model_file, int_output_dir)

    elif mode == "test":
        print("\n" + "=" * 60)
        print("Starting Testing")
        print("=" * 60)
        agent.test_controller()

    else:
        assert False, "Unsupported mode: {}".format(mode)

    return


def main(argv):
    args = load_args(argv)
    num_workers = args.parse_int("num_workers", 1)
    assert num_workers > 0

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
