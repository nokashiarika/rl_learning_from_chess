"""
Lesson 7: Multi-Feature State Representation
Solution - Reference implementation
"""

import chess
import numpy as np


def board_to_tensor(board):
    """
    Convert chess board to 8x8x12 tensor.
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = square // 8
            col = square % 8
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            tensor[row, col, channel] = 1.0
    
    return tensor


def piece_square_tables(board):
    """
    Calculate piece-square table values.
    """
    # Simple piece-square table: center squares are better
    pst = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    
    white_score = 0.0
    black_score = 0.0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = square // 8
            col = square % 8
            value = pst[row, col]
            if piece.color == chess.WHITE:
                white_score += value
            else:
                black_score += value
    
    return white_score - black_score


def mobility_and_safety(board):
    """
    Calculate mobility and king safety.
    """
    mobility = len(list(board.legal_moves))
    in_check = 1.0 if board.is_check() else 0.0
    return mobility, in_check


def extract_features(board):
    """
    Extract multi-feature vector from board.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    # Material count
    white_material = 0
    black_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    material = white_material - black_material
    
    # Piece-square table
    pst = piece_square_tables(board)
    
    # Mobility and safety
    mobility, in_check = mobility_and_safety(board)
    
    # Combine features
    features = np.array([
        material,
        pst,
        mobility,
        in_check
    ], dtype=np.float32)
    
    return features


def encode_state(board, method='hybrid'):
    """
    Main state encoding function.
    """
    if method == 'tensor':
        tensor = board_to_tensor(board)
        return tensor.flatten()
    elif method == 'features':
        return extract_features(board)
    elif method == 'hybrid':
        tensor = board_to_tensor(board).flatten()
        features = extract_features(board)
        return np.concatenate([tensor, features])
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    """
    Test the state representation implementation.
    """
    print("=== Multi-Feature State Representation Solution ===\n")
    
    board = chess.Board()
    
    # Test board tensor
    print("1. Testing board_to_tensor()...")
    tensor = board_to_tensor(board)
    print(f"   ✓ Tensor shape: {tensor.shape}")
    print(f"   ✓ Total values: {tensor.size}")
    print(f"   ✓ Non-zero values: {np.count_nonzero(tensor)}\n")
    
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
    
    print("=== Solution Complete ===")


if __name__ == "__main__":
    main()

