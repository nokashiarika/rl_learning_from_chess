"""
Lesson 3: Q-Learning from Scratch
Solution - Reference implementation
"""

import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for chess.
    """
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize Q-Learning agent.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def epsilon_greedy_action(self, state, legal_actions):
        """
        Choose action using epsilon-greedy policy.
        """
        if not legal_actions:
            return None
        
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Exploitation: action with highest Q-value
        # If state not seen before, all Q-values are 0 (default)
        best_action = max(legal_actions, key=lambda a: self.q_table[state][a])
        return best_action
    
    def update_q_value(self, state, action, reward, next_state, next_legal_actions):
        """
        Update Q-value using Bellman equation.
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Max future Q-value
        if next_legal_actions and next_state in self.q_table:
            max_future_q = max(
                (self.q_table[next_state][a] for a in next_legal_actions),
                default=0.0
            )
        else:
            max_future_q = 0.0
        
        # Q-Learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        target = reward + self.gamma * max_future_q
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
    
    def train_episode(self, env, max_steps=100):
        """
        Train the agent for one episode.
        """
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            # Get legal actions
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break
            
            # Choose action
            action = self.epsilon_greedy_action(state, legal_actions)
            if action is None:
                break
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Get next legal actions for Q-update
            next_legal_actions = env.get_legal_actions()
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, next_legal_actions)
            
            # Move to next state
            state = next_state
            
            # Check if done
            if done:
                break
        
        return total_reward, steps


def train_q_learning(env, agent, num_episodes=100):
    """
    Training loop for Q-Learning.
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
    Test Q-Learning implementation.
    """
    print("=== Q-Learning Solution ===\n")
    
    import sys
    import os
    import importlib.util
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, "lessons", "02_environment", "solution_environment.py")
    spec = importlib.util.spec_from_file_location("solution_environment", env_path)
    solution_environment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_environment)
    ChessEnvironment = solution_environment.ChessEnvironment
    
    env = ChessEnvironment()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # Test training
    print("Training for 20 episodes...")
    episode_rewards, episode_steps = train_q_learning(env, agent, num_episodes=20)
    
    print(f"\nFinal Results:")
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Average steps: {sum(episode_steps) / len(episode_steps):.2f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    
    print("\n=== Solution Complete ===")


if __name__ == "__main__":
    random.seed(42)
    main()

