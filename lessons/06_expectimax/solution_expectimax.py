"""
Lesson 6: Expectimax with Alpha-Beta Pruning
Solution - Reference implementation
"""

import chess
import math


def evaluate_position(board):
    """
    Evaluate a chess position from White's perspective.
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
    
    material_diff = white_material - black_material
    
    # Mobility bonus (small value per legal move)
    mobility_bonus = len(list(board.legal_moves)) * 0.01
    
    # King safety penalty (if in check)
    safety_penalty = 0.1 if board.is_check() else 0.0
    
    # From White's perspective
    if board.turn == chess.WHITE:
        score = material_diff + mobility_bonus - safety_penalty
    else:
        score = material_diff - mobility_bonus + safety_penalty
    
    return score


def minimax(board, depth, maximizing_player):
    """
    Minimax algorithm implementation.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval


def expectimax(board, depth, maximizing_player):
    """
    Expectimax algorithm for stochastic opponents.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = expectimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        # Expectation over opponent moves (stochastic)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return evaluate_position(board)
        
        total_eval = 0.0
        for move in legal_moves:
            board.push(move)
            eval = expectimax(board, depth - 1, True)
            board.pop()
            total_eval += eval
        
        # Return expected value (average)
        return total_eval / len(legal_moves)


def alpha_beta(board, depth, alpha, beta, maximizing_player):
    """
    Alpha-beta pruning optimization of minimax.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Prune!
        return max_eval
    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Prune!
        return min_eval


def choose_move(board, algorithm='minimax', depth=3):
    """
    Choose best move using specified algorithm.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0
    
    best_move = None
    best_score = -math.inf if board.turn == chess.WHITE else math.inf
    
    for move in legal_moves:
        board.push(move)
        
        if algorithm == 'minimax':
            eval = minimax(board, depth - 1, False)
        elif algorithm == 'expectimax':
            eval = expectimax(board, depth - 1, False)
        elif algorithm == 'alpha_beta':
            eval = alpha_beta(board, depth - 1, -math.inf, math.inf, False)
        else:
            board.pop()
            continue
        
        board.pop()
        
        # Update best move
        if board.turn == chess.WHITE:
            if eval > best_score:
                best_score = eval
                best_move = move
        else:
            if eval < best_score:
                best_score = eval
                best_move = move
    
    return best_move, best_score


def main():
    """
    Test the expectimax implementation.
    """
    print("=== Expectimax Solution ===\n")
    
    board = chess.Board()
    
    # Test evaluation
    print("1. Testing evaluation function...")
    score = evaluate_position(board)
    print(f"   ✓ Initial position score: {score:.2f}\n")
    
    # Test minimax
    print("2. Testing minimax (depth=2)...")
    best_move, best_score = choose_move(board, algorithm='minimax', depth=2)
    print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}\n")
    
    # Test expectimax
    print("3. Testing expectimax (depth=2)...")
    best_move, best_score = choose_move(board, algorithm='expectimax', depth=2)
    print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}\n")
    
    # Test alpha-beta
    print("4. Testing alpha-beta (depth=3)...")
    import time
    start = time.time()
    best_move, best_score = choose_move(board, algorithm='alpha_beta', depth=3)
    elapsed = time.time() - start
    print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}")
    print(f"   ✓ Time: {elapsed:.3f}s\n")
    
    # Compare speeds
    print("5. Comparing minimax vs alpha-beta speed (depth=3)...")
    start = time.time()
    choose_move(board, algorithm='minimax', depth=3)
    minimax_time = time.time() - start
    
    start = time.time()
    choose_move(board, algorithm='alpha_beta', depth=3)
    ab_time = time.time() - start
    
    print(f"   Minimax: {minimax_time:.3f}s")
    print(f"   Alpha-Beta: {ab_time:.3f}s")
    if ab_time > 0:
        print(f"   Speedup: {minimax_time/ab_time:.2f}x\n")
    
    print("=== Solution Complete ===")


if __name__ == "__main__":
    main()

