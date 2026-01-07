"""
Lesson 4: Gym Integration
Exercise - Wrap chess environment in Gym API
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class ChessGymEnv(gym.Env):
    """
    TODO 1: Inherit from gym.Env (gymnasium)
    
    This makes your environment compatible with Gymnasium and RL libraries.
    """
    # YOUR CODE HERE
    # Change the class definition to inherit from gym.Env
    pass
    
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
        
        # TODO 4: Define observation_space and action_space
        # 
        # Observation space: Since we use material count (single float), use Box
        #   - Box(low=-inf, high=inf, shape=(1,), dtype=np.float32)
        #
        # Action space: Discrete actions (move indices)
        #   - We'll use a large Discrete space (e.g., 218) to cover all possible moves
        #   - Discrete(218) - maximum number of legal moves in chess is 218
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(218)
        pass
    
    def reset(self, seed=None, options=None):
        """
        TODO 2: Reset the environment and return initial observation
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
        
        Returns:
            observation: Initial state observation
            info: Additional information (dict)
        
        Hint:
        - Call self.chess_env.reset() to get initial state
        - Return (observation, {}) - info can be empty dict
        """
        # YOUR CODE HERE
        state = self.chess_env.reset()
        observation = np.array([state], dtype=np.float32)
        return observation, {}
    
    def step(self, action):
        """
        TODO 3: Execute one step in the environment
        
        Args:
            action: Action to take (integer index)
        
        Returns:
            observation: Next state observation
            reward: Reward value
            terminated: True if episode ended (win/loss/draw)
            truncated: False (we don't use time limits)
            info: Additional information (dict)
        
        Hint:
        - Call self.chess_env.step(action) to get (state, reward, done, info)
        - Convert state to numpy array: np.array([state], dtype=np.float32)
        - terminated = done (episode ended)
        - truncated = False (no time limit)
        - Return (observation, reward, terminated, truncated, {})
        """
        # Get legal actions first
        legal_actions = self.chess_env.get_legal_actions()
        
        if not legal_actions:
            # No legal moves - game over
            obs = np.array([self.chess_env.get_state()], dtype=np.float32)
            return obs, 0.0, True, False, {}
        
        # Convert numpy types to Python int (Stable-Baselines3 passes numpy.int64)
        action = int(action)
        
        # Validate action: clamp to valid range (Option 3 logic)
        if action >= len(legal_actions):
            action = len(legal_actions) - 1  # Clamp to last valid action
        if action < 0:
            action = 0  # Clamp to first valid action
        
        # Now proceed with validated action
        state, reward, done, info = self.chess_env.step(action)
        observation = np.array([state], dtype=np.float32)
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        """
        Optional: Render the environment (not required for this exercise)
        """
        if self.chess_env.board is not None:
            print(self.chess_env.board)
        return None


def main():
    """
    Test the Gym environment
    """
    print("=== Gym Integration Exercise ===\n")
    
    env = ChessGymEnv()
    
    # Test observation and action spaces
    print("1. Testing spaces...")
    if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
        print(f"   ✓ Observation space: {env.observation_space}")
        print(f"   ✓ Action space: {env.action_space}\n")
    else:
        print("   ✗ Spaces not defined")
        return
    
    # Test reset
    print("2. Testing reset()...")
    try:
        obs, info = env.reset()
        if obs is not None:
            print(f"   ✓ Reset successful! Observation: {obs}, Shape: {obs.shape}")
            print(f"   Info: {info}\n")
        else:
            print("   ✗ reset() not implemented")
            return
    except Exception as e:
        print(f"   ✗ Error in reset(): {e}")
        return
    
    # Test step
    print("3. Testing step()...")
    try:
        # Sample from legal actions (Option 3 - most logical)
        legal_actions = env.chess_env.get_legal_actions()
        if legal_actions:
            action = random.choice(legal_actions)  # Always valid!
            obs, reward, terminated, truncated, info = env.step(action)
            if obs is not None:
                print(f"   ✓ Step successful!")
                print(f"   Observation: {obs}, Reward: {reward}")
                print(f"   Terminated: {terminated}, Truncated: {truncated}\n")
            else:
                print("   ✗ step() not implemented")
                return
        else:
            print("   ✗ No legal actions available")
            return
    except Exception as e:
        print(f"   ✗ Error in step(): {e}")
        return
    
    # Test multiple steps
    print("4. Testing multiple steps...")
    env.reset()
    for i in range(5):
        # Sample from legal actions (Option 3 - most logical)
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
    
    print("=== Exercise Complete ===")
    print("Run 'python -m pytest tests/test_lesson_04.py' to verify your implementation!")


if __name__ == "__main__":
    main()

