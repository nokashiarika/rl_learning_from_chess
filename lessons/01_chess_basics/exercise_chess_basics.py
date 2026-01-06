"""
Lesson 1: Chess Basics with Python
Exercise - Practice using python-chess library
"""

import chess


def create_chess_board():
    """
    TODO: Create a new chess board in the starting position
    
    Hint: Use chess.Board() to create a new board
    Returns: A chess.Board object
    """
    return chess.Board()


def get_legal_moves(board):
    """
    TODO: Get all legal moves for the current position
    
    Args:
        board: A chess.Board object
    
    Hint: Use board.legal_moves to get an iterator of legal moves
    Returns: A list of legal moves (convert the iterator to a list)
    """
    return list(board.legal_moves)


def make_move(board, move):
    """
    TODO: Make a move on the board
    
    Args:
        board: A chess.Board object
        move: A chess.Move object or a move in UCI notation (e.g., "e2e4")
    
    Hint: Use board.push() to apply a move
    Returns: The board after the move (board is modified in place)
    """
    board.push(move)
    return board


def check_game_status(board):
    """
    TODO: Check the current game status
    
    Args:
        board: A chess.Board object
    
    Returns: A dictionary with status information:
        - 'is_check': bool - Is the current player in check?
        - 'is_checkmate': bool - Is it checkmate?
        - 'is_stalemate': bool - Is it stalemate?
        - 'is_game_over': bool - Is the game finished?
    
    Hint: Use board.is_check(), board.is_checkmate(), etc.
    """
    return {
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_game_over': board.is_game_over()
    }

def main():
    """
    Main function to test your implementations
    """
    print("=== Chess Basics Exercise ===\n")
    
    # TODO: Create a board
    print("1. Creating a chess board...")
    board = create_chess_board()
    if board is not None:
        print("   ✓ Board created successfully!")
        print(f"   Current position:\n{board}\n")
    else:
        print("   ✗ Board creation not implemented yet")
        return
    
    # TODO: Get legal moves
    print("2. Getting legal moves...")
    legal_moves = get_legal_moves(board)
    if legal_moves is not None and len(legal_moves) > 0:
        print(f"   ✓ Found {len(legal_moves)} legal moves")
        print(f"   First 5 moves: {[str(move) for move in legal_moves[:5]]}\n")
    else:
        print("   ✗ Legal moves not implemented yet")
        return
    
    # TODO: Make a move
    print("3. Making a move (e2e4)...")
    if legal_moves:
        # Try to make the first legal move
        first_move = legal_moves[0]
        make_move(board, first_move)
        print(f"   ✓ Move {first_move} applied")
        print(f"   Position after move:\n{board}\n")
    
    # TODO: Check game status
    print("4. Checking game status...")
    status = check_game_status(board)
    if status is not None:
        print("   ✓ Game status retrieved:")
        for key, value in status.items():
            print(f"   - {key}: {value}")
    else:
        print("   ✗ Game status check not implemented yet")
    
    print("\n=== Exercise Complete ===")
    print("If all checks passed, you're ready for Lesson 2!")


if __name__ == "__main__":
    main()

