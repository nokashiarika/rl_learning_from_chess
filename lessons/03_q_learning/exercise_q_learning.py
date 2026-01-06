"""
Lesson 3: Q-Learning from Scratch
Exercise - Implement Q-Learning algorithm
"""

import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for chess.
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        TODO 1: Initialize Q-Learning agent
        
        Args:
            alpha: Learning rate (0 < alpha <= 1)
            gamma: Discount factor (0 <= gamma <= 1)
            epsilon: Exploration rate (0 <= epsilon <= 1)
        
        Create an empty Q-table to store Q(s,a) values.
        Hint: Use defaultdict(lambda: defaultdict(float)) for nested dictionary
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # YOUR CODE HERE
        # Initialize Q-table as a nested dictionary
        # Q-table structure: Q[state][action] = Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def epsilon_greedy_action(self, state, legal_actions):
        """
        TODO 2: Choose action using epsilon-greedy policy
        
        Args:
            state: Current state representation
            legal_actions: List of legal action indices
        
        Returns:
            action: Chosen action (index from legal_actions)
        
        Algorithm:
        1. With probability epsilon, return a random action (exploration)
        2. Otherwise, return the action with highest Q-value (exploitation)
        
        Hint:
        - Use random.random() < self.epsilon to decide exploration vs exploitation
        - For exploitation, find action with max Q-value: max(legal_actions, key=lambda a: self.q_table[state][a])
        - If state not in Q-table, default Q-value is 0
        """
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return max(legal_actions, key=lambda a: self.q_table[state][a])
    
    def update_q_value(self, state, action, reward, next_state, next_legal_actions):
        """
        TODO 3: Update Q-value using Bellman equation
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            next_legal_actions: Legal actions in next state
        
        Q-Learning update formula:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Steps:
        1. Get current Q-value: Q(s,a)
        2. Calculate max future Q-value: max(Q(s',a')) for all legal actions in next state
        3. Calculate target: r + γ*max(Q(s',a'))
        4. Update: Q(s,a) = Q(s,a) + α[target - Q(s,a)]
        
        Hint:
        - If next_state not in Q-table or no legal actions, max_future_q = 0
        - Use max() with a generator expression to find max Q-value
        """
        current_q_value = self.q_table[state][action]
        
        # Calculate max future Q-value (only for legal actions in next state)
        max_future_q_value = 0.0
        if next_legal_actions and next_state in self.q_table:
            max_future_q_value = max(
                (self.q_table[next_state][a] for a in next_legal_actions),
                default=0.0
            )
        
        # Bellman equation: target = r + γ*max(Q(s',a'))
        target = reward + self.gamma * max_future_q_value
        
        # Q-Learning update: Q(s,a) = Q(s,a) + α[target - Q(s,a)]
        self.q_table[state][action] = current_q_value + self.alpha * (target - current_q_value)

    
    def train_episode(self, env, max_steps=100):
        """
        TODO 4: Train the agent for one episode
        
        Args:
            env: ChessEnvironment instance
            max_steps: Maximum steps per episode
        
        Returns:
            total_reward: Sum of rewards in this episode
            steps: Number of steps taken
        
        Algorithm:
        1. Reset environment, get initial state
        2. For each step:
           a. Choose action using epsilon-greedy
           b. Take step in environment
           c. Update Q-value
           d. Move to next state
           e. If done, break
        3. Return total reward and steps
        """
        # YOUR CODE HERE
        # Hint:
        # - env.reset() returns initial state
        # - env.get_legal_actions() returns list of legal action indices
        # - env.step(action) returns (next_state, reward, done, info)
        # - Use self.epsilon_greedy_action() to choose action
        # - Use self.update_q_value() to update Q-table
        state = env.reset()
        total_reward = 0
        steps = 0
        for _ in range(max_steps):
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            action = self.epsilon_greedy_action(state, legal_actions)
            next_state, reward, done, _ = env.step(action)
            
            # Get legal actions for the next state (needed for Q-update)
            next_legal_actions = env.get_legal_actions()
            
            # Update Q-value using next state's legal actions
            self.update_q_value(state, action, reward, next_state, next_legal_actions)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break
        return total_reward, steps


def train_q_learning(env, agent, num_episodes=100):
    """
    Training loop for Q-Learning
    
    Args:
        env: ChessEnvironment instance
        agent: QLearningAgent instance
        num_episodes: Number of training episodes
    
    Returns:
        episode_rewards: List of total rewards per episode
        episode_steps: List of steps per episode
    """
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        total_reward, steps = agent.train_episode(env)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward (last 10): {avg_reward:.2f}, Steps: {steps}")
    
    return episode_rewards, episode_steps


def main():
    """
    Test Q-Learning implementation
    """
    print("=== Q-Learning Exercise ===\n")
    
    # Import environment (using importlib to handle numbered directories)
    import sys
    import os
    import importlib.util
    
    # Get the lessons directory (parent of current file's directory)
    lessons_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(lessons_dir, "02_environment", "exercise_environment.py")
    spec = importlib.util.spec_from_file_location("exercise_environment", env_path)
    exercise_environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exercise_environment)
    ChessEnvironment = exercise_environment.ChessEnvironment
    
    env = ChessEnvironment()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # Test Q-table initialization
    print("1. Testing Q-table initialization...")
    if hasattr(agent, 'q_table') and agent.q_table is not None:
        print("   ✓ Q-table initialized")
    else:
        print("   ✗ Q-table not initialized")
        return
    
    # Test epsilon-greedy
    print("2. Testing epsilon-greedy action selection...")
    env.reset()
    legal_actions = env.get_legal_actions()
    if legal_actions:
        state = env.get_state()
        action = agent.epsilon_greedy_action(state, legal_actions)
        if action is not None and action in legal_actions:
            print(f"   ✓ Action selected: {action}")
        else:
            print("   ✗ epsilon_greedy_action() not implemented correctly")
            return
    
    # Test Q-value update
    print("3. Testing Q-value update...")
    state = env.get_state()
    action = legal_actions[0] if legal_actions else None
    if action is not None:
        next_state, reward, done, _ = env.step(action)
        next_legal = env.get_legal_actions()
        agent.update_q_value(state, action, reward, next_state, next_legal)
        print(f"   ✓ Q-value updated")
    
    # Test training episode
    print("4. Testing training episode...")
    env.reset()
    total_reward, steps = agent.train_episode(env, max_steps=50)
    if total_reward is not None:
        print(f"   ✓ Episode completed! Reward: {total_reward}, Steps: {steps}")
    else:
        print("   ✗ train_episode() not implemented")
    
    print("\n=== Exercise Complete ===")
    print("Run 'python -m pytest tests/test_lesson_03.py' to verify your implementation!")


if __name__ == "__main__":
    random.seed(42)
    main()

