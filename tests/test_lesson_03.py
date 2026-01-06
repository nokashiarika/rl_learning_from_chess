"""
Checkpoint tests for Lesson 3: Q-Learning from Scratch
Run these tests to verify your Q-Learning implementation.
"""

import pytest
import sys
import os
import importlib.util

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import QLearningAgent from numbered directory using importlib
q_learning_path = os.path.join(base_dir, "lessons", "03_q_learning", "exercise_q_learning.py")
spec = importlib.util.spec_from_file_location("exercise_q_learning", q_learning_path)
exercise_q_learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exercise_q_learning)
QLearningAgent = exercise_q_learning.QLearningAgent

# Import ChessEnvironment from solution
env_path = os.path.join(base_dir, "lessons", "02_environment", "solution_environment.py")
spec = importlib.util.spec_from_file_location("solution_environment", env_path)
solution_environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution_environment)
ChessEnvironment = solution_environment.ChessEnvironment


def test_q_table_initialization():
    """Test that Q-table is initialized."""
    agent = QLearningAgent()
    
    assert hasattr(agent, 'q_table'), "Agent should have a q_table attribute"
    assert agent.q_table is not None, "q_table should not be None"


def test_epsilon_greedy_action():
    """Test epsilon-greedy action selection."""
    agent = QLearningAgent(epsilon=0.0)  # Pure exploitation
    env = ChessEnvironment()
    env.reset()
    
    legal_actions = env.get_legal_actions()
    if legal_actions:
        state = env.get_state()
        action = agent.epsilon_greedy_action(state, legal_actions)
        
        assert action is not None, "Should return an action"
        assert action in legal_actions, "Action should be legal"


def test_q_value_update():
    """Test Q-value update."""
    agent = QLearningAgent()
    env = ChessEnvironment()
    env.reset()
    
    state = env.get_state()
    legal_actions = env.get_legal_actions()
    if legal_actions:
        action = legal_actions[0]
        
        # Store initial Q-value FIRST (before any modifications)
        initial_q = agent.q_table[state][action]
        assert initial_q == 0.0, "Initial Q-value should be 0.0"
        
        # Take step to get next state
        next_state, reward, done, _ = env.step(action)
        next_legal = env.get_legal_actions()
        
        # Pre-populate Q-table for next state to ensure Q-value will update
        # This simulates a scenario where we've already learned some Q-values
        if next_legal:
            # Use an action that's different from the current action to avoid
            # overwriting if state == next_state (which can happen with material count)
            pre_pop_action = next_legal[0]
            if state == next_state and pre_pop_action == action and len(next_legal) > 1:
                # If same state and same action, use a different action
                pre_pop_action = next_legal[1]
            agent.q_table[next_state][pre_pop_action] = 0.5
        
        # Update Q-value - should see max_future_q = 0.5
        agent.update_q_value(state, action, reward, next_state, next_legal)
        
        # Check that Q-value changed
        updated_q = agent.q_table[state][action]
        assert updated_q != initial_q, "Q-value should update when future Q-value is non-zero"
        
        # Verify the update follows Bellman equation correctly
        # Expected calculation:
        # target = reward + gamma * max_future_q = 0 + 0.9 * 0.5 = 0.45
        # new_q = initial_q + alpha * (target - initial_q) = 0 + 0.1 * (0.45 - 0) = 0.045
        expected_q = 0.0 + 0.1 * (0.0 + 0.9 * 0.5 - 0.0)  # 0.045
        assert abs(updated_q - expected_q) < 0.001, \
            f"Expected Q-value ~{expected_q:.3f}, got {updated_q:.3f}. Bellman equation may be incorrect."
        assert updated_q > 0, "Q-value should be positive after update with positive future value"


def test_train_episode():
    """Test that training episode runs."""
    agent = QLearningAgent()
    env = ChessEnvironment()
    
    total_reward, steps = agent.train_episode(env, max_steps=10)
    
    assert total_reward is not None, "Should return total reward"
    assert steps is not None, "Should return number of steps"
    assert isinstance(total_reward, (int, float)), "Reward should be a number"
    assert isinstance(steps, int), "Steps should be an integer"
    assert steps >= 0, "Steps should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

