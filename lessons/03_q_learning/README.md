# Lesson 3: Q-Learning from Scratch

## Learning Objectives

By the end of this lesson, you will:
- Understand the Q-Learning algorithm
- Implement Q-value updates using the Bellman equation
- Implement epsilon-greedy exploration
- Build a complete Q-Learning training loop

## Theory: Q-Learning

Q-Learning is a value-based reinforcement learning algorithm that learns the optimal action-value function Q(s,a), which represents the expected cumulative reward of taking action `a` in state `s`.

### Key Concepts

#### Q-Values
- Q(s,a) = Expected future reward from state s, taking action a, then following optimal policy
- Higher Q-values indicate better actions

#### Bellman Equation
The Q-value update follows:
```
Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate (0 < α ≤ 1) - how much to update
- **γ (gamma)**: Discount factor (0 ≤ γ ≤ 1) - importance of future rewards
- **r**: Immediate reward
- **s'**: Next state
- **max(Q(s',a'))**: Best Q-value in next state

#### Epsilon-Greedy Exploration
- **Exploration**: Try random actions to discover better strategies
- **Exploitation**: Use the best known action (highest Q-value)
- **Epsilon (ε)**: Probability of exploration (0 ≤ ε ≤ 1)
  - With probability ε: choose random action
  - With probability (1-ε): choose action with highest Q-value

#### Q-Table
- Dictionary/matrix storing Q(s,a) for each state-action pair
- For large state spaces, we use function approximation (neural networks) - that's DQN!

## Exercise Instructions

Open `exercise_q_learning.py` and implement:
1. Q-table initialization
2. Epsilon-greedy action selection
3. Q-value update using Bellman equation
4. Training loop that runs episodes

Start simple - we'll improve the implementation as we go!

