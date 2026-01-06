"""
State representation utilities for chess.

This module provides helper functions for encoding chess board states.
You'll build these functions as you progress through the lessons.
"""

import chess


def material_count(board):
    """
    Calculate material count difference (White - Black).
    
    Args:
        board: chess.Board object
    
    Returns:
        Material difference as a float
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
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
    
    return float(white_material - black_material)

