"""
Checkpoint tests for Lesson 2: Building the RL Environment
Run these tests to verify your environment implementation.
"""

import pytest
import sys
import os
import importlib.util

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import from numbered directory using importlib
env_path = os.path.join(base_dir, "lessons", "02_environment", "exercise_environment.py")
spec = importlib.util.spec_from_file_location("exercise_environment", env_path)
exercise_environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exercise_environment)
ChessEnvironment = exercise_environment.ChessEnvironment


def test_environment_reset():
    """Test that reset() initializes the board and returns a state."""
    env = ChessEnvironment()
    state = env.reset()
    
    assert state is not None, "reset() should return a state"
    assert isinstance(state, (int, float)), "State should be a number"
    assert env.board is not None, "Board should be initialized"


def test_get_state():
    """Test that get_state() returns a valid state representation."""
    env = ChessEnvironment()
    env.reset()
    state = env.get_state()
    
    assert state is not None, "get_state() should return a state"
    assert isinstance(state, (int, float)), "State should be a number"


def test_get_reward():
    """Test that get_reward() returns a valid reward."""
    env = ChessEnvironment()
    env.reset()
    reward = env.get_reward()
    
    assert reward is not None, "get_reward() should return a reward"
    assert isinstance(reward, (int, float)), "Reward should be a number"
    assert -1 <= reward <= 1, "Reward should be between -1 and 1"


def test_step():
    """Test that step() executes an action and returns valid results."""
    env = ChessEnvironment()
    env.reset()
    
    legal_actions = env.get_legal_actions()
    if legal_actions:
        action = legal_actions[0]
        state, reward, done, info = env.step(action)
        
        assert state is not None, "step() should return a state"
        assert isinstance(state, (int, float)), "State should be a number"
        assert isinstance(reward, (int, float)), "Reward should be a number"
        assert isinstance(done, bool), "done should be a boolean"
        assert isinstance(info, dict), "info should be a dictionary"


def test_multiple_steps():
    """Test that multiple steps work correctly."""
    env = ChessEnvironment()
    env.reset()
    
    for _ in range(5):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        action = legal_actions[0]
        state, reward, done, info = env.step(action)
        
        assert state is not None
        if done:
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

