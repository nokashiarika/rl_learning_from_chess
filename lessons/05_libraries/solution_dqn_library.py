"""
Lesson 5: Using RL Libraries
Solution - Reference implementation
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def create_environment():
    """
    Create and return the Gym environment.
    """
    import sys
    import os
    import importlib.util
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gym_wrapper_path = os.path.join(base_dir, "04_gym_integration", "solution_gym_wrapper.py")
    spec = importlib.util.spec_from_file_location("solution_gym_wrapper", gym_wrapper_path)
    solution_gym_wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_gym_wrapper)
    ChessGymEnv = solution_gym_wrapper.ChessGymEnv
    
    return ChessGymEnv()


def train_dqn_model(env, total_timesteps=10000):
    """
    Create and train a DQN model.
    """
    # Create DQN model
    model = DQN(
        'MlpPolicy',  # Multi-Layer Perceptron policy
        env,
        verbose=1,  # Show training progress
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=1000,
        train_freq=4,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    return model


def evaluate_model(model, env, n_episodes=10):
    """
    Evaluate the trained model.
    """
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_episodes,
        deterministic=True
    )
    return mean_reward, std_reward


def compare_with_q_learning(dqn_rewards, q_learning_rewards):
    """
    Compare DQN with Q-Learning results.
    """
    dqn_mean = np.mean(dqn_rewards)
    dqn_std = np.std(dqn_rewards)
    ql_mean = np.mean(q_learning_rewards)
    ql_std = np.std(q_learning_rewards)
    
    print(f"DQN Results:")
    print(f"  Mean reward: {dqn_mean:.2f} ± {dqn_std:.2f}")
    print(f"Q-Learning Results:")
    print(f"  Mean reward: {ql_mean:.2f} ± {ql_std:.2f}")
    
    if dqn_mean > ql_mean:
        print(f"\n✓ DQN performs better (difference: {dqn_mean - ql_mean:.2f})")
    elif ql_mean > dqn_mean:
        print(f"\n✓ Q-Learning performs better (difference: {ql_mean - dqn_mean:.2f})")
    else:
        print("\n≈ Both methods perform similarly")


def main():
    """
    Main training and evaluation script.
    """
    print("=== DQN Library Solution ===\n")
    
    # Create environment
    print("1. Creating environment...")
    env = create_environment()
    print("   ✓ Environment created\n")
    
    # Train model
    print("2. Training DQN model (this may take a few minutes)...")
    model = train_dqn_model(env, total_timesteps=5000)
    print("   ✓ Training complete\n")
    
    # Evaluate model
    print("3. Evaluating model...")
    mean_reward, std_reward = evaluate_model(model, env, n_episodes=10)
    print(f"   ✓ Evaluation complete!")
    print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
    
    # Compare with Q-Learning (placeholder)
    print("4. Comparison with Q-Learning...")
    q_learning_rewards = [0.0] * 10  # Placeholder - would run Q-Learning agent
    dqn_rewards = [mean_reward] * 10
    compare_with_q_learning(dqn_rewards, q_learning_rewards)
    
    print("\n=== Solution Complete ===")
    print("Libraries make RL much easier, but understanding the fundamentals")
    print("(like Q-Learning) helps you use libraries effectively!")


if __name__ == "__main__":
    main()

