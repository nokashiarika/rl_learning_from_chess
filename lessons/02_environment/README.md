# Lesson 2: Building the RL Environment

## Learning Objectives

By the end of this lesson, you will:
- Understand the components of an RL environment
- Implement state representation for chess
- Design reward functions
- Create a complete RL environment interface

## Theory: RL Environment Components

A reinforcement learning environment needs:
1. **State (Observation)**: A representation of the current game situation
2. **Action Space**: The set of possible actions the agent can take
3. **Reward Function**: Feedback signal indicating how good an action was
4. **Transition Function**: How the environment changes when an action is taken

### State Representation

Chess has an enormous state space (~10^43 positions). For learning, we need to simplify:
- **Material Count**: Sum of piece values (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9)
- **Piece Positions**: Where pieces are located
- **Game Status**: Check, checkmate, stalemate

We'll start simple with material count, then expand later.

### Reward Design

Rewards guide learning:
- **Win**: +1 (agent wins)
- **Loss**: -1 (agent loses)
- **Draw**: 0
- **Ongoing**: 0 (or small intermediate rewards)

### Environment Interface

Standard RL environment methods:
- `reset()`: Initialize a new episode, return initial state
- `step(action)`: Apply action, return (next_state, reward, done, info)
- `get_state()`: Encode current board position
- `get_reward()`: Calculate reward based on game outcome

## Exercise Instructions

Open `exercise_environment.py` and implement:
1. `reset()` - Initialize board
2. `step(action)` - Apply move and return results
3. `get_state()` - Encode board as a state representation
4. `get_reward()` - Calculate reward

Start with simple state representation (material count) - we'll improve it later!

