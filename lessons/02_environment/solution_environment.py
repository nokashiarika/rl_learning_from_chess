"""
Lesson 2: Building the RL Environment
Solution - Reference implementation
"""

import chess
import random
import numpy as np


class ChessEnvironment:
    """
    A simple chess environment for reinforcement learning.
    
    The agent plays as White. The opponent (Black) makes random moves.
    """
    
    def __init__(self):
        """
        Initialize the environment.
        """
        self.board = None
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King value not used in material count
        }
    
    def reset(self):
        """
        Reset the environment to initial state.
        """
        self.board = chess.Board()
        return self.get_state()
    
    def get_state(self):
        """
        Encode the current board position as a state.
        
        Uses material count difference (White - Black).
        """
        if self.board is None:
            return 0.0
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return float(white_material - black_material)
    
    def get_reward(self):
        """
        Calculate the reward for the current position.
        """
        if self.board is None:
            return 0.0
        
        if not self.board.is_game_over():
            return 0.0
        
        # Check if White won (Black is checkmated)
        if self.board.is_checkmate() and self.board.turn == chess.BLACK:
            return 1.0
        
        # Check if White lost (White is checkmated)
        if self.board.is_checkmate() and self.board.turn == chess.WHITE:
            return -1.0
        
        # Draw (stalemate or other draw condition)
        return 0.0
    
    def step(self, action):
        """
        Apply an action and return the result.
        """
        if self.board is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # Get legal moves
        legal_moves = list(self.board.legal_moves)
        
        if not legal_moves:
            # No legal moves - game should be over
            state = self.get_state()
            reward = self.get_reward()
            done = True
            return state, reward, done, {}
        
        # Apply agent's move (White)
        # Handle both Python int and numpy integer types
        if isinstance(action, (int, np.integer)):
            action = int(action)  # Convert to Python int
            if action >= len(legal_moves):
                raise ValueError(f"Action {action} out of range. {len(legal_moves)} legal moves available.")
            move = legal_moves[action]
        else:
            move = action
        
        self.board.push(move)
        
        # If it's Black's turn, make a random move
        if self.board.turn == chess.BLACK and not self.board.is_game_over():
            black_legal_moves = list(self.board.legal_moves)
            if black_legal_moves:
                black_move = random.choice(black_legal_moves)
                self.board.push(black_move)
        
        # Get new state and reward
        state = self.get_state()
        reward = self.get_reward()
        done = self.board.is_game_over()
        
        return state, reward, done, {}
    
    def get_legal_actions(self):
        """
        Get list of legal actions (moves) for current position.
        """
        if self.board is None:
            return []
        return list(range(len(list(self.board.legal_moves))))


def main():
    """
    Test the environment implementation
    """
    print("=== Chess Environment Solution ===\n")
    
    env = ChessEnvironment()
    
    # Test reset
    print("1. Testing reset()...")
    initial_state = env.reset()
    print(f"   ✓ Reset successful! Initial state: {initial_state}")
    print(f"   Board:\n{env.board}\n")
    
    # Test get_state
    print("2. Testing get_state()...")
    state = env.get_state()
    print(f"   ✓ State retrieved: {state}")
    
    # Test get_reward
    print("3. Testing get_reward()...")
    reward = env.get_reward()
    print(f"   ✓ Reward calculated: {reward}")
    
    # Test step
    print("4. Testing step()...")
    legal_actions = env.get_legal_actions()
    if legal_actions:
        next_state, reward, done, info = env.step(legal_actions[0])
        print(f"   ✓ Step successful!")
        print(f"   Next state: {next_state}, Reward: {reward}, Done: {done}")
    
    print("\n=== Solution Complete ===")


if __name__ == "__main__":
    random.seed(42)
    main()

