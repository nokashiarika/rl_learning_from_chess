# Lesson 5: Using RL Libraries

## Learning Objectives

By the end of this lesson, you will:
- Use stable-baselines3 to train RL agents
- Understand how libraries abstract complexity
- Compare library-based training with custom Q-Learning
- Visualize training progress

## Theory: Library Abstractions

### Why Use Libraries?

Libraries like stable-baselines3 provide:
- **Pre-implemented algorithms**: DQN, PPO, A2C, etc.
- **Optimized implementations**: Efficient, tested code
- **Easy hyperparameter tuning**: Built-in support
- **Visualization tools**: TensorBoard integration
- **Model saving/loading**: Easy persistence

### DQN (Deep Q-Network)

DQN extends Q-Learning by using neural networks to approximate Q-values:
- Handles large state spaces
- Learns complex patterns
- More powerful than tabular Q-Learning

### stable-baselines3 API

```python
from stable_baselines3 import DQN

# Create model
model = DQN('MlpPolicy', env, verbose=1)

# Train
model.learn(total_timesteps=10000)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
```

## Exercise Instructions

Open `exercise_dqn_library.py` and implement:
1. Create environment using your Gym wrapper
2. Initialize DQN model
3. Train the model
4. Evaluate and compare with your Q-Learning implementation

See how much simpler it is with libraries!

