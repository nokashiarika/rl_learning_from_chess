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
    pass


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
    pass


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
    # YOUR CODE HERE
    pass


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
    pass


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

