"""
Lesson 4: Gym Integration
Solution - Reference implementation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class ChessGymEnv(gym.Env):
    """
    Chess environment wrapped in Gym API.
    """
    
    def __init__(self):
        """
        Initialize the Gym environment.
        """
        super().__init__()
        
        # Import the chess environment (using importlib to handle numbered directories)
        import sys
        import os
        import importlib.util
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base_dir, "02_environment", "solution_environment.py")
        spec = importlib.util.spec_from_file_location("solution_environment", env_path)
        solution_environment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_environment)
        ChessEnvironment = solution_environment.ChessEnvironment
        
        self.chess_env = ChessEnvironment()
        
        # Define observation and action spaces
        # Observation: Material count (single float)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Action: Discrete move index (max 218 legal moves in chess)
        self.action_space = spaces.Discrete(218)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment and return initial observation.
        """
        if seed is not None:
            np.random.seed(seed)
        
        state = self.chess_env.reset()
        observation = np.array([state], dtype=np.float32)
        return observation, {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        """
        # Get legal actions first
        legal_actions = self.chess_env.get_legal_actions()
        
        if not legal_actions:
            # No legal moves - game over
            obs = np.array([self.chess_env.get_state()], dtype=np.float32)
            return obs, 0.0, True, False, {}
        
        # Convert numpy types to Python int (Stable-Baselines3 passes numpy.int64)
        action = int(action)
        
        # Validate action: clamp to valid range
        if action >= len(legal_actions):
            action = len(legal_actions) - 1  # Clamp to last valid action
        if action < 0:
            action = 0  # Clamp to first valid action
        
        # Now proceed with validated action
        state, reward, done, _ = self.chess_env.step(action)
        observation = np.array([state], dtype=np.float32)
        
        # Gym API: terminated = episode ended, truncated = time limit
        terminated = done
        truncated = False
        
        return observation, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        """
        Render the environment.
        """
        if self.chess_env.board is not None:
            print(self.chess_env.board)
        return None


def main():
    """
    Test the Gym environment.
    """
    print("=== Gym Integration Solution ===\n")
    
    env = ChessGymEnv()
    
    # Test spaces
    print("1. Testing spaces...")
    print(f"   ✓ Observation space: {env.observation_space}")
    print(f"   ✓ Action space: {env.action_space}\n")
    
    # Test reset
    print("2. Testing reset()...")
    obs, info = env.reset()
    print(f"   ✓ Reset successful! Observation: {obs}, Shape: {obs.shape}")
    print(f"   Info: {info}\n")
    
    # Test step
    print("3. Testing step()...")
    # Sample from legal actions (most logical approach)
    legal_actions = env.chess_env.get_legal_actions()
    if legal_actions:
        action = random.choice(legal_actions)  # Always valid!
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step successful!")
        print(f"   Observation: {obs}, Reward: {reward}")
        print(f"   Terminated: {terminated}, Truncated: {truncated}\n")
    else:
        print("   ✗ No legal actions available")
    
    # Test multiple steps
    print("4. Testing multiple steps...")
    env.reset()
    for i in range(5):
        # Sample from legal actions (most logical approach)
        legal_actions = env.chess_env.get_legal_actions()
        if not legal_actions:
            print(f"   Episode ended at step {i+1} (no legal moves)")
            break
        action = random.choice(legal_actions)  # Always valid!
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            break
    print("   ✓ Multiple steps successful!\n")
    
    print("=== Solution Complete ===")


if __name__ == "__main__":
    main()

