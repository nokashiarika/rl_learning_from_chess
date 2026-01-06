# Lesson 1: Chess Basics with Python

## Learning Objectives

By the end of this lesson, you will:
- Understand how to use the `python-chess` library
- Create and manipulate chess boards
- Generate and work with legal moves
- Check game status (check, checkmate, draw, etc.)

## Theory: Chess Representation

In reinforcement learning, we need to represent the chess game state. The `python-chess` library provides a convenient way to:
- Represent the board state
- Generate legal moves
- Check game outcomes
- Make moves

## Key Concepts

### Board Creation
- `chess.Board()` creates a new board in the starting position
- The board tracks the current position, turn, castling rights, etc.

### Legal Moves
- `board.legal_moves` returns an iterator of all legal moves
- Moves are represented as `chess.Move` objects

### Making Moves
- `board.push(move)` applies a move to the board
- `board.pop()` undoes the last move (useful for search algorithms)

### Game Status
- `board.is_check()` - Is the current player in check?
- `board.is_checkmate()` - Is it checkmate?
- `board.is_stalemate()` - Is it stalemate?
- `board.is_game_over()` - Is the game finished?

## Exercise Instructions

Open `exercise_chess_basics.py` and complete the TODOs:
1. Create a chess board
2. Generate and display legal moves
3. Make a move and check the game status

Try to implement these yourself before checking the solution!

