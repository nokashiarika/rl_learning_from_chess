"""
Lesson 2: Building the RL Environment
Exercise - Implement a chess RL environment
"""

import chess


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
        TODO 1: Reset the environment to initial state
        
        Initialize a new chess board and return the initial state.
        
        Returns:
            state: The initial state representation
        """
        # YOUR CODE HERE
        # Hint: Create a new chess.Board() and return get_state()
        self.board = chess.Board()
        return self.get_state()
    
    def get_state(self):
        """
        TODO 3: Encode the current board position as a state
        
        For now, use a simple material count representation:
        - Calculate material difference (White pieces - Black pieces)
        - Return this as a single number
        
        Future improvements: Add piece positions, game status, etc.
        
        Returns:
            state: A representation of the current board state
                 For now, just the material count difference (float)
        """
        # YOUR CODE HERE
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
        TODO 4: Calculate the reward for the current position
        
        Reward function:
        - +1 if White wins (checkmate for Black)
        - -1 if White loses (checkmate for White)
        - 0 if draw (stalemate or other draw condition)
        - 0 if game is ongoing
        
        Returns:
            reward: The reward value (float)
        """
        # YOUR CODE HERE
        # Hint:
        # - Check if game is over: self.board.is_game_over()
        # - Check if White won: self.board.is_checkmate() and self.board.turn == chess.BLACK
        # - Check if White lost: self.board.is_checkmate() and self.board.turn == chess.WHITE
        # - Check for draw: self.board.is_stalemate() or other draw conditions
        if self.board.is_game_over():
            if self.board.is_checkmate() and self.board.turn == chess.BLACK:
                return 1.0
            elif self.board.is_checkmate() and self.board.turn == chess.WHITE:
                return -1.0
            else:
                return 0.0
        return 0.0
    
    def step(self, action):
        """
        TODO 2: Apply an action and return the result
        
        Steps:
        1. Get all legal moves
        2. Apply the action (move) to the board
        3. If it's Black's turn, make a random move for the opponent
        4. Get the new state and reward
        5. Check if episode is done
        
        Args:
            action: An integer index into the list of legal moves, or a chess.Move object
        
        Returns:
            state: New state after the action
            reward: Reward for this step
            done: Whether the episode is finished
            info: Additional information (dict)
        """
        # YOUR CODE HERE
        # Hint:
        # - Get legal moves: list(self.board.legal_moves)
        # - If action is an integer, use it as an index: legal_moves[action]
        # - Apply move: self.board.push(move)
        # - If it's Black's turn, make a random move (use random.choice)
        # - Get new state: self.get_state()
        # - Get reward: self.get_reward()
        # - Check if done: self.board.is_game_over()
        # - Return (state, reward, done, {})
        legal_moves = list(self.board.legal_moves)
        if isinstance(action, int):
            if action >= len(legal_moves):
                raise ValueError(f"Action {action} out of range. {len(legal_moves)} legal moves available.")
            move = legal_moves[action]
        else:
            move = action
        self.board.push(move)
        return self.get_state(), self.get_reward(), self.board.is_game_over(), {}
    def get_legal_actions(self):
        """
        Get list of legal actions (moves) for current position.
        Returns list of move indices.
        """
        if self.board is None:
            return []
        return list(range(len(list(self.board.legal_moves))))


def main():
    """
    Test the environment implementation
    """
    print("=== Chess Environment Exercise ===\n")
    
    env = ChessEnvironment()
    
    # Test reset
    print("1. Testing reset()...")
    initial_state = env.reset()
    if initial_state is not None:
        print(f"   ✓ Reset successful! Initial state: {initial_state}")
        print(f"   Board:\n{env.board}\n")
    else:
        print("   ✗ reset() not implemented yet")
        return
    
    # Test get_state
    print("2. Testing get_state()...")
    state = env.get_state()
    if state is not None:
        print(f"   ✓ State retrieved: {state}")
    else:
        print("   ✗ get_state() not implemented yet")
        return
    
    # Test get_reward
    print("3. Testing get_reward()...")
    reward = env.get_reward()
    if reward is not None:
        print(f"   ✓ Reward calculated: {reward}")
    else:
        print("   ✗ get_reward() not implemented yet")
        return
    
    # Test step
    print("4. Testing step()...")
    legal_actions = env.get_legal_actions()
    if legal_actions:
        next_state, reward, done, info = env.step(legal_actions[0])
        if next_state is not None:
            print(f"   ✓ Step successful!")
            print(f"   Next state: {next_state}, Reward: {reward}, Done: {done}")
        else:
            print("   ✗ step() not implemented yet")
    else:
        print("   ✗ No legal actions available")
    
    print("\n=== Exercise Complete ===")
    print("Run 'python -m pytest tests/test_lesson_02.py' to verify your implementation!")


if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible random moves
    main()

