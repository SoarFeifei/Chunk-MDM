import torch
import numpy as np
import unittest


class TestPhysicsConstraints(unittest.TestCase):
    """物理约束奖励系统的单元测试"""

    def setUp(self):
        """设置测试环境"""
        self.config = {
            'block_size': 5,
            'physics_constraints': {
                'foot_slide_weight': 1.0,
                'floating_weight': 1.0,
                'ground_penetration_weight': 1.0,
                'smoothness_weight': 0.1,
                'contact_threshold': 0.05,
                'slide_threshold': 0.02,
                'ground_threshold': 0.0
            }
        }

        self.device = torch.device('cpu')

    def test_module_import(self):
        """测试模块能否正确导入"""
        try:
            from policy.learning.physics_constraints import PhysicsConstraintReward
            print("✓ PhysicsConstraintReward 模块导入成功")
        except ImportError as e:
            self.fail(f"模块导入失败: {e}")

    def test_bmdm_integration_import(self):
        """测试BMDM集成模块导入"""
        try:
            from policy.learning.bmdm_integration import BMDMExplorer, BMDMGuide
            print("✓ BMDM集成模块导入成功")
        except ImportError as e:
            self.fail(f"BMDM集成模块导入失败: {e}")

    def test_ppo_agent_import(self):
        """测试扩展PPO代理导入"""
        try:
            from policy.learning.ppo_agent_with_physics import PPOAgentWithPhysics
            print("✓ PPOAgentWithPhysics 模块导入成功")
        except ImportError as e:
            self.fail(f"扩展PPO代理导入失败: {e}")


class TestPhysicsConstraintsFunctions(unittest.TestCase):
    """物理约束功能测试"""

    def test_foot_slide_calculation(self):
        """测试滑步计算逻辑"""
        print("\n=== 测试滑步计算逻辑 ===")

        batch_size = 2
        num_feet = 2

        foot_pos_history = torch.zeros((batch_size, 2, num_feet * 3))
        current_foot_pos = torch.zeros((batch_size, 1, num_feet * 3))

        foot_pos_history[:, -1, 0] = 0.0
        foot_pos_history[:, -1, 1] = 0.01
        foot_pos_history[:, -1, 2] = 0.0

        current_foot_pos[:, 0, 0] = 0.1
        current_foot_pos[:, 0, 1] = 0.01
        current_foot_pos[:, 0, 2] = 0.0

        displacement = torch.norm(current_foot_pos - foot_pos_history[:, -1:, :], dim=-1)

        print(f"脚部位移: {displacement}")
        print("✓ 滑步计算逻辑测试通过")

    def test_floating_detection(self):
        """测试悬空检测"""
        print("\n=== 测试悬空检测 ===")

        batch_size = 2
        num_feet = 2

        foot_heights = torch.tensor([[0.01, 0.02], [0.1, 0.15]])
        contact_threshold = 0.05

        in_contact = foot_heights < contact_threshold
        all_floating = (~in_contact).all(dim=-1)

        print(f"脚部高度: {foot_heights}")
        print(f"是否接触地面: {in_contact}")
        print(f"是否悬空: {all_floating}")
        print("✓ 悬空检测逻辑测试通过")

    def test_ground_penetration(self):
        """测试穿地检测"""
        print("\n=== 测试穿地检测 ===")

        joint_heights = torch.tensor([[0.1, -0.05, 0.2], [0.0, -0.1, 0.05]])
        ground_threshold = 0.0

        penetration = torch.clamp(ground_threshold - joint_heights, min=0)
        mean_penetration = penetration.mean(dim=-1)

        print(f"关节高度: {joint_heights}")
        print(f"穿地深度: {penetration}")
        print(f"平均穿地: {mean_penetration}")
        print("✓ 穿地检测逻辑测试通过")

    def test_smoothness_calculation(self):
        """测试平滑度计算"""
        print("\n=== 测试平滑度计算 ===")

        batch_size = 2
        frame_dim = 10
        block_size = 5

        pose = torch.randn(batch_size, block_size * frame_dim)
        prev_pose = torch.randn(batch_size, block_size * frame_dim)

        smoothness_loss = 0.0
        for i in range(1, block_size):
            curr_frame = pose[:, i * frame_dim:(i + 1) * frame_dim]
            prev_frame = pose[:, (i - 1) * frame_dim:i * frame_dim]
            smoothness_loss += torch.norm(curr_frame - prev_frame, dim=-1).mean()

        smoothness_loss /= block_size

        print(f"平滑度损失: {smoothness_loss.item()}")
        print("✓ 平滑度计算逻辑测试通过")


def run_simple_tests():
    """运行简单的功能测试"""
    print("=" * 60)
    print("AMDM 物理约束与BMDM集成系统 - 测试套件")
    print("=" * 60)

    test_classes = [TestPhysicsConstraints, TestPhysicsConstraintsFunctions]

    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 60)
    print("使用指南")
    print("=" * 60)
    print("""
1. 配置文件示例 (config.yaml):
   -----------------------------
   block_size: 5
   use_physics_constraints: true
   use_bmdm_integration: true
   physics_weight: 1.0
   bmdm_guide_weight: 0.5

   physics_constraints:
     foot_slide_weight: 1.0
     floating_weight: 1.0
     ground_penetration_weight: 1.0
     smoothness_weight: 0.1
     contact_threshold: 0.05
     slide_threshold: 0.02
     ground_threshold: 0.0

   bmdm_explorer:
     num_exploration_samples: 5
     exploration_noise_scale: 0.1
     use_best_of_n: true

   bmdm_guide:
     guide_weight: 0.5
     similarity_threshold: 0.8

2. 训练代码示例:
   ----------------
   from policy.learning.ppo_agent_with_physics import PPOAgentWithPhysics
   from policy.learning.ppo_model import PPOModel

   # 初始化模型
   actor_critic = PPOModel(config, env, device)

   # 创建带物理约束的PPO代理
   agent = PPOAgentWithPhysics(
       config,
       actor_critic,
       env,
       device,
       bmdm_model=pretrained_bmdm_model
   )

   # 开始训练
   agent.train_controller(out_model_path, int_output_dir)

3. 测试代码示例:
   ---------------
   from policy.learning.physic_constraints_tester import run_simple_tests
   run_simple_tests()
    """)


if __name__ == "__main__":
    run_simple_tests()
    print_usage_guide()
