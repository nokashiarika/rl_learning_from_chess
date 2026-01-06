"""
Lesson 1: Chess Basics with Python
Solution - Reference implementation
"""

import chess


def create_chess_board():
    """
    Create a new chess board in the starting position
    """
    return chess.Board()


def get_legal_moves(board):
    """
    Get all legal moves for the current position
    """
    return list(board.legal_moves)


def make_move(board, move):
    """
    Make a move on the board
    """
    # If move is a string, convert it to a Move object
    if isinstance(move, str):
        move = chess.Move.from_uci(move)
    
    board.push(move)
    return board


def check_game_status(board):
    """
    Check the current game status
    """
    return {
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_game_over': board.is_game_over()
    }


def main():
    """
    Main function to test implementations
    """
    print("=== Chess Basics Solution ===\n")
    
    # Create a board
    print("1. Creating a chess board...")
    board = create_chess_board()
    print("   ✓ Board created successfully!")
    print(f"   Current position:\n{board}\n")
    
    # Get legal moves
    print("2. Getting legal moves...")
    legal_moves = get_legal_moves(board)
    print(f"   ✓ Found {len(legal_moves)} legal moves")
    print(f"   First 5 moves: {[str(move) for move in legal_moves[:5]]}\n")
    
    # Make a move
    print("3. Making a move (e2e4)...")
    if legal_moves:
        first_move = legal_moves[0]
        make_move(board, first_move)
        print(f"   ✓ Move {first_move} applied")
        print(f"   Position after move:\n{board}\n")
    
    # Check game status
    print("4. Checking game status...")
    status = check_game_status(board)
    print("   ✓ Game status retrieved:")
    for key, value in status.items():
        print(f"   - {key}: {value}")
    
    print("\n=== Solution Complete ===")


if __name__ == "__main__":
    main()

