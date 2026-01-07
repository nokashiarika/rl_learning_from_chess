# Lesson 4: Gym Integration

## Learning Objectives

By the end of this lesson, you will:
- Understand the Gymnasium API standard
- Wrap a custom environment in the Gymnasium interface
- Use Gymnasium's space definitions for observations and actions
- Understand the benefits of standardization

## Theory: Gymnasium API

Gymnasium (formerly OpenAI Gym) provides a standard interface for RL environments. This standardization:
- Makes environments interchangeable
- Enables use of RL libraries (stable-baselines3, etc.)
- Provides consistent testing and evaluation

### Gymnasium Environment Interface

A Gymnasium environment must implement:
1. **observation_space**: Defines the shape and type of observations
2. **action_space**: Defines the shape and type of actions
3. **reset()**: Returns (observation, info)
4. **step(action)**: Returns (observation, reward, terminated, truncated, info)

### Gymnasium Spaces

Common space types:
- **Discrete(n)**: Integer actions from 0 to n-1
- **Box(low, high, shape)**: Continuous values in a box
- **MultiDiscrete([n1, n2, ...])**: Multiple discrete values

### Return Values

- **observation**: Current state representation
- **reward**: Scalar reward value
- **terminated**: True if episode ended (win/loss/draw)
- **truncated**: True if episode ended due to time limit
- **info**: Additional information (dict)

## Exercise Instructions

Open `exercise_gym_wrapper.py` and implement:
1. Inherit from `gymnasium.Env` (import as `import gymnasium as gym`)
2. Define `observation_space` and `action_space`
3. Implement `reset()` returning (observation, info)
4. Implement `step(action)` returning (observation, reward, terminated, truncated, info)

This will make your environment compatible with RL libraries!

