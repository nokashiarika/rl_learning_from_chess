"""
Lesson 7: Multi-Feature State Representation
Exercise - Implement advanced state encoding with matrix operations
"""

import chess
import numpy as np


def board_to_tensor(board):
    """
    TODO 1: Convert chess board to 8x8x12 tensor
    
    Create a 3D tensor where:
    - First two dimensions: 8x8 chess board
    - Third dimension: 12 channels (one per piece type/color)
    
    Channels:
    0: White Pawn
    1: White Knight
    2: White Bishop
    3: White Rook
    4: White Queen
    5: White King
    6: Black Pawn
    7: Black Knight
    8: Black Bishop
    9: Black Rook
    10: Black Queen
    11: Black King
    
    Args:
        board: chess.Board object
    
    Returns:
        tensor: numpy array of shape (8, 8, 12), dtype=np.float32
    
    Hint:
    - Initialize tensor with zeros: np.zeros((8, 8, 12), dtype=np.float32)
    - Loop through chess.SQUARES (0-63)
    - Convert square to (row, col): row = square // 8, col = square % 8
    - Get piece: board.piece_at(square)
    - Map piece to channel based on piece_type and color
    - Set tensor[row, col, channel] = 1.0
    """
    # YOUR CODE HERE
    
    # Initialize tensor
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    # TODO: Create mapping from (piece_type, color) to channel
    # piece_to_channel = {
    #     (chess.PAWN, chess.WHITE): 0,
    #     (chess.KNIGHT, chess.WHITE): 1,
    #     ...
    # }
    
    # TODO: Loop through all squares
    # for square in chess.SQUARES:
    #     piece = board.piece_at(square)
    #     if piece is not None:
    #         row = square // 8
    #         col = square % 8
    #         channel = piece_to_channel[(piece.piece_type, piece.color)]
    #         tensor[row, col, channel] = 1.0
    
    return tensor


def piece_square_tables(board):
    """
    TODO 4: Calculate piece-square table values
    
    Piece-square tables give positional bonuses based on where pieces are.
    Center squares are generally better than edges.
    
    Args:
        board: chess.Board object
    
    Returns:
        pst_score: Float, positional score (positive = good for White)
    
    Hint:
    - Create simple piece-square table (8x8 array)
    - Center squares (d4, d5, e4, e5) have higher values
    - For each piece, sum its square table value
    - White pieces: add value, Black pieces: subtract value
    """
    # YOUR CODE HERE
    
    # TODO: Create simple piece-square table
    # Center squares (rows 3-4, cols 3-4) have value 1.0
    # Edge squares have value 0.0
    # pst = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 1, 0, 0],  # Center rows
    #     [0, 0, 1, 1, 1, 1, 0, 0],
    #     ...
    # ])
    
    # TODO: Sum piece-square values for all pieces
    # white_score = sum(pst[row, col] for white pieces)
    # black_score = sum(pst[row, col] for black pieces)
    # return white_score - black_score
    
    return 0.0


def mobility_and_safety(board):
    """
    TODO 5: Calculate mobility and king safety features
    
    Mobility: Number of legal moves (more moves = better)
    King Safety: Is the king in check? (in check = bad)
    
    Args:
        board: chess.Board object
    
    Returns:
        mobility: Float, number of legal moves
        in_check: Float, 1.0 if in check, 0.0 otherwise
    
    Hint:
    - mobility = len(list(board.legal_moves))
    - in_check = 1.0 if board.is_check() else 0.0
    """
    # YOUR CODE HERE
    
    # TODO: Calculate mobility
    # mobility = len(list(board.legal_moves))
    
    # TODO: Calculate king safety
    # in_check = 1.0 if board.is_check() else 0.0
    
    return 0.0, 0.0


def extract_features(board):
    """
    TODO 2: Extract multi-feature vector from board
    
    Extract various features that capture different aspects of the position:
    - Material count
    - Piece-square table values
    - Mobility
    - King safety
    - (Optional: pawn structure, center control, etc.)
    
    Args:
        board: chess.Board object
    
    Returns:
        features: numpy array of shape (n_features,), dtype=np.float32
    
    Hint:
    - Material: white_material - black_material
    - Piece-square: call piece_square_tables(board)
    - Mobility/Safety: call mobility_and_safety(board)
    - Concatenate all into one array: np.array([material, pst, mobility, in_check])
    """
    # YOUR CODE HERE
    
    # TODO: Calculate material count
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    # white_material = sum(...)
    # black_material = sum(...)
    # material = white_material - black_material
    
    # TODO: Get piece-square table score
    # pst = piece_square_tables(board)
    
    # TODO: Get mobility and safety
    # mobility, in_check = mobility_and_safety(board)
    
    # TODO: Combine into feature vector
    # features = np.array([material, pst, mobility, in_check], dtype=np.float32)
    
    # Return features
    return np.array([0.0], dtype=np.float32)  # Placeholder


def encode_state(board, method='hybrid'):
    """
    TODO 3: Main state encoding function
    
    Encode board state using specified method:
    - 'tensor': 8x8x12 board tensor (flattened to 768)
    - 'features': Multi-feature vector
    - 'hybrid': Concatenate both (tensor + features)
    
    Args:
        board: chess.Board object
        method: 'tensor', 'features', or 'hybrid'
    
    Returns:
        state: numpy array ready for neural network input
    
    Hint:
    - If method == 'tensor':
    #   tensor = board_to_tensor(board)  # (8, 8, 12)
    #   return tensor.flatten()  # (768,)
    - If method == 'features':
    #   return extract_features(board)  # (n_features,)
    - If method == 'hybrid':
    #   tensor = board_to_tensor(board).flatten()  # (768,)
    #   features = extract_features(board)  # (n_features,)
    #   return np.concatenate([tensor, features])  # (768 + n_features,)
    """
    # YOUR CODE HERE
    
    if method == 'tensor':
        # TODO: Return flattened board tensor
        pass
    elif method == 'features':
        # TODO: Return feature vector
        pass
    elif method == 'hybrid':
        # TODO: Return concatenated tensor + features
        pass
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return np.array([0.0], dtype=np.float32)  # Placeholder


def main():
    """
    Test the state representation implementation
    """
    print("=== Multi-Feature State Representation Exercise ===\n")
    
    board = chess.Board()
    
    # Test board tensor
    print("1. Testing board_to_tensor()...")
    tensor = board_to_tensor(board)
    if tensor.shape == (8, 8, 12):
        print(f"   ✓ Tensor shape: {tensor.shape}")
        print(f"   ✓ Total values: {tensor.size}")
        print(f"   ✓ Non-zero values: {np.count_nonzero(tensor)}\n")
    else:
        print(f"   ✗ Wrong shape: {tensor.shape}, expected (8, 8, 12)\n")
        return
    
    # Test feature extraction
    print("2. Testing extract_features()...")
    features = extract_features(board)
    print(f"   ✓ Feature vector shape: {features.shape}")
    print(f"   ✓ Features: {features}\n")
    
    # Test encoding methods
    print("3. Testing encode_state() methods...")
    
    tensor_state = encode_state(board, method='tensor')
    print(f"   ✓ Tensor encoding shape: {tensor_state.shape}")
    
    feature_state = encode_state(board, method='features')
    print(f"   ✓ Feature encoding shape: {feature_state.shape}")
    
    hybrid_state = encode_state(board, method='hybrid')
    print(f"   ✓ Hybrid encoding shape: {hybrid_state.shape}\n")
    
    # Compare sizes
    print("4. Comparing encoding sizes...")
    print(f"   Tensor: {tensor_state.size} values")
    print(f"   Features: {feature_state.size} values")
    print(f"   Hybrid: {hybrid_state.size} values")
    print(f"   Original (material only): 1 value\n")
    
    print("=== Exercise Complete ===")
    print("Next: Update environment to use new state encoding!")


if __name__ == "__main__":
    main()

