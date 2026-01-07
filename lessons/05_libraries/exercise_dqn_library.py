"""
Lesson 5: Using RL Libraries
Exercise - Train DQN using stable-baselines3
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


def create_environment():
    """
    TODO 1: Create and return your Gym environment
    
    Import and instantiate your ChessGymEnv from lesson 04.
    
    Returns:
        env: ChessGymEnv instance
    """
    # YOUR CODE HERE
    # Hint: Import from lessons.04_gym_integration.solution_gym_wrapper
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
    TODO 2: Create and train a DQN model
    
    Args:
        env: Gym environment
        total_timesteps: Number of training timesteps
    
    Returns:
        model: Trained DQN model
    
    Steps:
    1. Create DQN model: DQN('MlpPolicy', env, verbose=1)
    2. Train the model: model.learn(total_timesteps=total_timesteps)
    3. Return the model
    
    Hint:
    - 'MlpPolicy' means Multi-Layer Perceptron (neural network)
    - verbose=1 shows training progress
    """
    # YOUR CODE HERE
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(model, env, n_episodes=10):
    """
    TODO 3: Evaluate the trained model
    
    Args:
        model: Trained DQN model
        env: Gym environment
        n_episodes: Number of evaluation episodes
    
    Returns:
        mean_reward: Average reward across episodes
        std_reward: Standard deviation of rewards
    
    Hint: Use evaluate_policy from stable_baselines3.common.evaluation
    """
    return evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True)


def compare_with_q_learning(dqn_rewards, q_learning_rewards):
    """
    TODO 4: Compare DQN with Q-Learning results
    
    Args:
        dqn_rewards: List of rewards from DQN evaluation
        q_learning_rewards: List of rewards from Q-Learning
    
    Print comparison statistics:
    - Mean rewards
    - Standard deviations
    - Which performs better
    """
    # YOUR CODE HERE
    # Hint: Use np.mean() and np.std() for statistics
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
    Main training and evaluation script
    """
    print("=== DQN Library Exercise ===\n")
    
    # TODO 1: Create environment
    print("1. Creating environment...")
    env = create_environment()
    if env is None:
        print("   ✗ Environment creation not implemented")
        return
    print("   ✓ Environment created\n")
    
    # TODO 2: Train model
    print("2. Training DQN model (this may take a few minutes)...")
    model = train_dqn_model(env, total_timesteps=5000)
    if model is None:
        print("   ✗ Model training not implemented")
        return
    print("   ✓ Training complete\n")
    
    # TODO 3: Evaluate model
    print("3. Evaluating model...")
    mean_reward, std_reward = evaluate_model(model, env, n_episodes=10)
    if mean_reward is not None:
        print(f"   ✓ Evaluation complete!")
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
    else:
        print("   ✗ Evaluation not implemented")
        return
    
    # TODO 4: Compare with Q-Learning (optional)
    print("4. Comparison with Q-Learning...")
    # For this exercise, we'll use dummy Q-Learning rewards
    # In practice, you'd run your Q-Learning agent and collect rewards
    q_learning_rewards = [0.0] * 10  # Placeholder
    dqn_rewards = [mean_reward] * 10  # Placeholder
    compare_with_q_learning(dqn_rewards, q_learning_rewards)
    
    print("\n=== Exercise Complete ===")
    print("Notice how much simpler it is to use libraries!")
    print("However, understanding the fundamentals (Q-Learning) helps you")
    print("understand what the library is doing under the hood.")


if __name__ == "__main__":
    main()

