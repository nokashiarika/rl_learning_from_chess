"""
Checkpoint tests for Lesson 4: Gym Integration
Run these tests to verify your Gym wrapper implementation.
"""

import pytest
import sys
import os
import numpy as np
import importlib.util

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import from numbered directory using importlib
gym_wrapper_path = os.path.join(base_dir, "lessons", "04_gym_integration", "exercise_gym_wrapper.py")
spec = importlib.util.spec_from_file_location("exercise_gym_wrapper", gym_wrapper_path)
exercise_gym_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exercise_gym_wrapper)
ChessGymEnv = exercise_gym_wrapper.ChessGymEnv


def test_gym_env_inheritance():
    """Test that environment inherits from gym.Env."""
    env = ChessGymEnv()
    
    # Check inheritance
    from gym import Env
    assert isinstance(env, Env), "Should inherit from gym.Env"


def test_observation_space():
    """Test that observation_space is defined."""
    env = ChessGymEnv()
    
    assert hasattr(env, 'observation_space'), "Should have observation_space"
    assert env.observation_space is not None, "observation_space should not be None"


def test_action_space():
    """Test that action_space is defined."""
    env = ChessGymEnv()
    
    assert hasattr(env, 'action_space'), "Should have action_space"
    assert env.action_space is not None, "action_space should not be None"


def test_reset():
    """Test that reset() returns correct format."""
    env = ChessGymEnv()
    obs, info = env.reset()
    
    assert obs is not None, "Should return observation"
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert isinstance(info, dict), "Info should be a dictionary"
    assert obs.shape == (1,), "Observation should have shape (1,)"


def test_step():
    """Test that step() returns correct format."""
    env = ChessGymEnv()
    env.reset()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs is not None, "Should return observation"
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert isinstance(reward, (int, float)), "Reward should be a number"
    assert isinstance(terminated, bool), "terminated should be boolean"
    assert isinstance(truncated, bool), "truncated should be boolean"
    assert isinstance(info, dict), "Info should be a dictionary"


def test_multiple_steps():
    """Test that multiple steps work correctly."""
    env = ChessGymEnv()
    obs, info = env.reset()
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        if terminated or truncated:
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

