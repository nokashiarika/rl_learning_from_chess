# RL Chess Learning Project - Guided Learning Experience

## Overview

A **guided, hands-on learning project** that teaches reinforcement learning through chess. You'll implement code step-by-step with scaffolded exercises, hints, and checkpoints. Each module includes:

- **Exercise files** with TODOs and hints
- **Solution files** for reference (check after attempting)
- **Learning guides** explaining concepts
- **Checkpoint tests** to verify your implementation

## Teaching Approach

This project uses a **scaffolded learning** approach:

1. **You Write the Code**: Exercise files contain function signatures and TODOs - you fill in the implementation
2. **Hints When Needed**: Comments provide guidance without giving away the answer
3. **Test Your Work**: Checkpoint tests verify your implementation works correctly
4. **Learn from Solutions**: After attempting, compare with solution files to see alternative approaches
5. **Progressive Complexity**: Each lesson builds on previous concepts

**Key Principle**: You learn by doing, not by reading completed code. The scaffolded structure ensures you understand each component while building it yourself.

## Project Structure

```
RL practice/
├── requirements.txt                    # Python dependencies
├── README.md                           # Main learning guide with roadmap
├── lessons/                            # Step-by-step learning modules
│   ├── 01_chess_basics/
│   │   ├── README.md                   # Understanding chess in Python
│   │   ├── exercise_chess_basics.py    # Practice exercises
│   │   └── solution_chess_basics.py    # Reference solution
│   ├── 02_environment/
│   │   ├── README.md                   # Building RL environment
│   │   ├── exercise_environment.py     # Implement environment
│   │   └── solution_environment.py     # Reference solution
│   ├── 03_q_learning/
│   │   ├── README.md                   # Q-Learning theory & practice
│   │   ├── exercise_q_learning.py      # Implement Q-Learning
│   │   └── solution_q_learning.py      # Reference solution
│   ├── 04_gym_integration/
│   │   ├── README.md                   # Gym API introduction
│   │   ├── exercise_gym_wrapper.py     # Wrap environment in Gym
│   │   └── solution_gym_wrapper.py     # Reference solution
│   └── 05_libraries/
│       ├── README.md                   # Using stable-baselines3
│       ├── exercise_dqn_library.py     # Implement with library
│       └── solution_dqn_library.py     # Reference solution
├── src/                                # Your working implementation
│   ├── __init__.py
│   ├── chess_env/
│   │   ├── __init__.py
│   │   ├── chess_environment.py        # Build this in lesson 02
│   │   └── chess_gym_env.py            # Build this in lesson 04
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── q_learning_scratch.py       # Build this in lesson 03
│   │   └── dqn_stable_baselines.py     # Build this in lesson 05
│   └── utils/
│       ├── __init__.py
│       ├── state_representation.py     # Helper functions
│       └── visualization.py            # Plotting utilities
├── tests/                              # Checkpoint tests
│   ├── __init__.py
│   ├── test_lesson_02.py               # Test your environment
│   ├── test_lesson_03.py               # Test your Q-Learning
│   └── test_lesson_04.py               # Test your Gym wrapper
├── notebooks/                          # Interactive learning
│   ├── 01_chess_exploration.ipynb     # Explore python-chess
│   ├── 02_rl_concepts.ipynb           # RL fundamentals
│   └── 03_training_analysis.ipynb      # Analyze training results
└── solutions/                          # Complete reference solutions
    └── (reference implementations)
```

## Getting Started: Project Setup (Your First Exercise!)

**Location**: This section in README.md

Before diving into the lessons, let's set up your development environment. This is your first hands-on exercise!

### Setup Exercise

**Your Tasks**:

1. **TODO 1: Create a Python virtual environment**
   - Hint: Use `python3 -m venv venv` or `python -m venv venv`
   - Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
   - **Why?** Virtual environments keep project dependencies isolated

2. **TODO 2: Install required packages**
   - Create a `requirements.txt` file with these dependencies:
     - `python-chess` - Chess board representation and move generation
     - `numpy` - Numerical computations
     - `matplotlib` - Plotting and visualization
     - `gymnasium` - RL environment interface (formerly Gym)
     - `shimmy` - Compatibility layer for Gym environments
     - `stable-baselines3` - RL algorithms library
     - `torch` - PyTorch for neural networks (used by stable-baselines3)
     - `pytest` - Testing framework
     - `jupyter` - For interactive notebooks
   - Install with: `pip install -r requirements.txt`
   - **Why?** These libraries provide the tools we'll use throughout the project

3. **TODO 3: Verify installation**
   - Try importing each package: `python -c "import chess; import numpy; import gymnasium; print('All packages installed!')"`
   - **Why?** Verifying ensures everything is set up correctly

4. **TODO 4: Create project structure**
   - Create the directories shown in the Project Structure section above
   - You can do this manually or write a small Python script to create them
   - **Why?** Good project organization makes development easier

**Checkpoint**: Once you can import all packages and your directory structure matches the plan, you're ready for Lesson 1!

**Learning**: Understanding Python project setup, virtual environments, and dependency management are essential skills for any Python developer.

---

## Learning Path (Guided Exercises)

### Lesson 1: Chess Basics with Python

**File**: `lessons/01_chess_basics/exercise_chess_basics.py`

- **Your Task**: 
  - TODO: Create a chess board using python-chess
  - TODO: Generate and print legal moves
  - TODO: Make a move and check game status
- **Hints**: Use `chess.Board()`, `board.legal_moves`, `board.push()`
- **Learning**: Understanding chess representation, move generation
- **Checkpoint**: Complete all TODOs, run solution comparison
- **Solution**: Check `lessons/01_chess_basics/solution_chess_basics.py` after attempting

### Lesson 2: Building the RL Environment

**File**: `lessons/02_environment/exercise_environment.py`

- **Your Task**:
  - TODO 1: Implement `reset()` method (initialize board)
  - TODO 2: Implement `step(action)` method (apply move, return reward)
  - TODO 3: Implement `get_state()` method (encode board state)
  - TODO 4: Implement `get_reward()` method (calculate reward)
- **Hints**: 
  - Reward: +1 for win, -1 for loss, 0 for draw/ongoing
  - State: Start simple (material count), we'll improve later
- **Learning**: RL environment interface, state/action/reward design
- **Checkpoint**: Run `python -m pytest tests/test_lesson_02.py`
- **Solution**: Check `lessons/02_environment/solution_environment.py` after attempting

### Lesson 3: Q-Learning from Scratch

**File**: `lessons/03_q_learning/exercise_q_learning.py`

- **Your Task**:
  - TODO 1: Initialize Q-table (dictionary: state -> action -> Q-value)
  - TODO 2: Implement `epsilon_greedy_action(state, epsilon)` 
  - TODO 3: Implement `update_q_value(state, action, reward, next_state, alpha, gamma)`
  - TODO 4: Implement training loop (episodes, steps, Q-updates)
- **Hints**:
  - Q-update formula: `Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]`
  - Epsilon-greedy: random action with prob ε, else best action
- **Learning**: Q-Learning algorithm, Bellman equation, exploration vs exploitation
- **Checkpoint**: Run `python -m pytest tests/test_lesson_03.py`
- **Solution**: Check `lessons/03_q_learning/solution_q_learning.py` after attempting

### Lesson 4: Gym Integration

**File**: `lessons/04_gym_integration/exercise_gym_wrapper.py`

- **Your Task**:
  - TODO 1: Inherit from `gymnasium.Env` (import as `import gymnasium as gym`)
  - TODO 2: Implement `reset()` returning (observation, info)
  - TODO 3: Implement `step(action)` returning (observation, reward, terminated, truncated, info)
  - TODO 4: Define `observation_space` and `action_space`
- **Hints**: 
  - Use `gymnasium.spaces.Discrete` for action space
  - Use `gymnasium.spaces.Box` or `gymnasium.spaces.MultiDiscrete` for observations
- **Learning**: Gymnasium API conventions, standardization benefits
- **Checkpoint**: Run `python -m pytest tests/test_lesson_04.py`
- **Solution**: Check `lessons/04_gym_integration/solution_gym_wrapper.py` after attempting

### Lesson 5: Using RL Libraries

**File**: `lessons/05_libraries/exercise_dqn_library.py`

- **Your Task**:
  - TODO 1: Create environment using your Gym wrapper
  - TODO 2: Initialize DQN model from stable-baselines3
  - TODO 3: Train the model with `.learn()`
  - TODO 4: Evaluate and compare with your Q-Learning implementation
- **Hints**: 
  - `from stable_baselines3 import DQN`
  - `model = DQN('MlpPolicy', env)`
  - `model.learn(total_timesteps=10000)`
- **Learning**: Library abstractions, when to use libraries vs custom code
- **Checkpoint**: Train model, visualize learning curves
- **Solution**: Check `lessons/05_libraries/solution_dqn_library.py` after attempting

## Key Learning Objectives

1. **Understanding RL Fundamentals**:
   - States, actions, rewards
   - Q-values and Bellman equation
   - Exploration vs exploitation

2. **Chess-Specific Challenges**:
   - Large state space (requires state abstraction)
   - Action space complexity (legal move generation)
   - Reward shaping (intermediate rewards)

3. **Library Usage**:
   - Gym API conventions
   - Stable-baselines3 abstractions
   - When to use libraries vs custom code

## Technical Considerations

- **State Space Reduction**: Chess has ~10^43 possible positions. We'll use simplified representations:
  - Material count (piece values)
  - Piece-square tables
  - Board features (castling rights, en passant)
  
- **Action Space**: Use `python-chess` to generate legal moves, encode as discrete actions

- **Reward Design**: 
  - Win: +1, Loss: -1, Draw: 0
  - Optional: Piece capture bonuses, check bonuses

- **Training Strategy**: Start with simplified chess (fewer pieces) to make learning tractable

## How to Use This Project

1. **Start with Setup**: Complete the setup exercise in this README (create venv, install packages, verify)
2. **Begin Lesson 1**: Read the lesson README, complete the exercise
3. **Try First**: Implement the TODOs in the exercise file
4. **Test**: Run checkpoint tests to verify your implementation
5. **Compare**: Check the solution file if stuck (but try first!)
6. **Move On**: Once tests pass, proceed to next lesson
7. **Ask Questions**: Use the learning guides and notebooks for deeper understanding

## Learning Progression

- **Setup Exercise**: Project setup and environment - ~30-60 minutes
- **Lesson 1**: Chess basics - ~2-3 hours
- **Lesson 2**: Environment building - ~3-4 hours
- **Lesson 3**: Q-Learning implementation - ~4-6 hours
- **Lesson 4**: Gym integration - ~2-3 hours
- **Lesson 5**: Library usage - ~2-3 hours
- **Total**: ~15-20 hours of guided learning

Each lesson builds on the previous, so complete them in order!
