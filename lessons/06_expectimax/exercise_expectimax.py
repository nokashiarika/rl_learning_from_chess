"""
Lesson 6: Expectimax with Alpha-Beta Pruning
Exercise - Implement game tree search algorithms
"""

import chess
import math


def evaluate_position(board):
    """
    TODO 1: Evaluate a chess position
    
    Calculate a score for the current position from White's perspective.
    Positive = good for White, Negative = good for Black.
    
    Features to include:
    - Material count (piece values)
    - Piece-square tables (positional bonuses)
    - Mobility (number of legal moves)
    - King safety (is king in check?)
    
    Args:
        board: chess.Board object
    
    Returns:
        score: Float evaluation (positive favors White)
    
    Hint:
    - Piece values: Pawn=1, Knight/Bishop=3, Rook=5, Queen=9
    - Count material for both sides
    - Add small bonus for mobility (legal moves count)
    - Penalty if king is in check
    """
    # YOUR CODE HERE
    # Start with material count
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
    
    # TODO: Calculate material for both sides
    # Hint: Loop through chess.SQUARES, get piece_at(square), check color
    
    # TODO: Add mobility bonus (small value like 0.01 per legal move)
    # Hint: len(list(board.legal_moves))
    
    # TODO: Add king safety penalty (if in check, subtract small value)
    # Hint: board.is_check()
    
    # Return: material_difference + mobility_bonus - safety_penalty
    pass


def minimax(board, depth, maximizing_player):
    """
    TODO 2: Implement minimax algorithm
    
    Recursively evaluate all moves to a certain depth.
    Maximizing player (White) tries to maximize score.
    Minimizing player (Black) tries to minimize score.
    
    Args:
        board: chess.Board object
        depth: How many moves ahead to look
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        best_score: Best evaluation found at this depth
    
    Algorithm:
    1. Base case: depth == 0 or game_over -> return evaluate_position(board)
    2. If maximizing_player:
       - Try all legal moves
       - Return maximum of minimax results
    3. Else (minimizing):
       - Try all legal moves
       - Return minimum of minimax results
    
    Hint:
    - Use board.legal_moves to get moves
    - Use board.push(move) to make move, board.pop() to undo
    - Remember to undo moves after evaluating!
    """
    # YOUR CODE HERE
    
    # Base case: depth 0 or game over
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    # TODO: If maximizing player (White)
    # - Initialize max_eval = -math.inf
    # - For each legal move:
    #   - Make the move (board.push)
    #   - Recursively call minimax with depth-1, maximizing_player=False
    #   - Undo the move (board.pop)
    #   - Update max_eval = max(max_eval, eval)
    # - Return max_eval
    
    # TODO: Else (minimizing player - Black)
    # - Initialize min_eval = math.inf
    # - For each legal move:
    #   - Make the move
    #   - Recursively call minimax with depth-1, maximizing_player=True
    #   - Undo the move
    #   - Update min_eval = min(min_eval, eval)
    # - Return min_eval
    
    pass


def expectimax(board, depth, maximizing_player):
    """
    TODO 3: Implement expectimax algorithm
    
    Similar to minimax, but for stochastic (random) opponents.
    Instead of minimizing, takes the expected value of opponent moves.
    
    Args:
        board: chess.Board object
        depth: How many moves ahead to look
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        expected_score: Expected evaluation
    
    Algorithm:
    1. Base case: same as minimax
    2. If maximizing_player:
       - Return max of expectimax results (same as minimax)
    3. Else (stochastic opponent):
       - Return average (expected value) of expectimax results
       - expected = sum(all_results) / len(all_results)
    
    Hint:
    - Maximizing part is same as minimax
    - Minimizing part becomes expectation (average)
    """
    # YOUR CODE HERE
    
    # Base case
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    # TODO: If maximizing player
    # - Same as minimax maximizing part
    
    # TODO: Else (stochastic opponent)
    # - Get all legal moves
    # - Calculate sum of expectimax results
    # - Return average: sum / len(moves)
    
    pass


def alpha_beta(board, depth, alpha, beta, maximizing_player):
    """
    TODO 4: Implement alpha-beta pruning
    
    Optimized minimax that prunes branches that can't improve the result.
    
    Args:
        board: chess.Board object
        depth: How many moves ahead to look
        alpha: Best value maximizer can guarantee (starts at -inf)
        beta: Best value minimizer can guarantee (starts at +inf)
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        best_score: Best evaluation found
    
    Algorithm:
    1. Base case: same as minimax
    2. If maximizing_player:
       - Try moves, update alpha = max(alpha, eval)
       - If alpha >= beta, prune (return alpha)
    3. Else:
       - Try moves, update beta = min(beta, eval)
       - If beta <= alpha, prune (return beta)
    
    Hint:
    - Alpha-beta pruning: if alpha >= beta, we can stop searching
    - This happens when we find a move so good that opponent won't allow it
    """
    # YOUR CODE HERE
    
    # Base case
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    
    # TODO: If maximizing player
    # - max_eval = -math.inf
    # - For each move:
    #   - Make move
    #   - eval = alpha_beta(..., alpha, beta, False)
    #   - Undo move
    #   - max_eval = max(max_eval, eval)
    #   - alpha = max(alpha, eval)
    #   - if alpha >= beta: break (prune!)
    # - Return max_eval
    
    # TODO: Else (minimizing)
    # - min_eval = math.inf
    # - For each move:
    #   - Make move
    #   - eval = alpha_beta(..., alpha, beta, True)
    #   - Undo move
    #   - min_eval = min(min_eval, eval)
    #   - beta = min(beta, eval)
    #   - if beta <= alpha: break (prune!)
    # - Return min_eval
    
    pass


def choose_move(board, algorithm='minimax', depth=3):
    """
    TODO 5: Choose best move using specified algorithm
    
    Args:
        board: chess.Board object
        algorithm: 'minimax', 'expectimax', or 'alpha_beta'
        depth: Search depth
    
    Returns:
        best_move: chess.Move object (best move found)
        best_score: Evaluation of that move
    
    Algorithm:
    1. Get all legal moves
    2. For each move:
       - Make the move
       - Evaluate using chosen algorithm
       - Undo the move
       - Track best move
    3. Return best move and score
    
    Hint:
    - For minimax: call minimax(result, depth-1, False)
    - For expectimax: call expectimax(result, depth-1, False)
    - For alpha_beta: call alpha_beta(result, depth-1, -inf, +inf, False)
    - Remember: after making move, it's opponent's turn (maximizing_player=False)
    """
    # YOUR CODE HERE
    
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0
    
    best_move = None
    best_score = -math.inf if board.turn == chess.WHITE else math.inf
    
    # TODO: For each legal move:
    #   - Make the move (board.push)
    #   - Evaluate based on algorithm:
    #     * minimax: minimax(board, depth-1, False)
    #     * expectimax: expectimax(board, depth-1, False)
    #     * alpha_beta: alpha_beta(board, depth-1, -math.inf, math.inf, False)
    #   - Undo move (board.pop)
    #   - Update best_move and best_score
    #   - For White: maximize, for Black: minimize
    
    # TODO: Return best_move, best_score
    
    pass


def main():
    """
    Test the expectimax implementation
    """
    print("=== Expectimax Exercise ===\n")
    
    board = chess.Board()
    
    # Test evaluation function
    print("1. Testing evaluation function...")
    score = evaluate_position(board)
    print(f"   ✓ Initial position score: {score:.2f}\n")
    
    # Test minimax
    print("2. Testing minimax (depth=2)...")
    best_move, best_score = choose_move(board, algorithm='minimax', depth=2)
    if best_move:
        print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}\n")
    else:
        print("   ✗ minimax not implemented")
        return
    
    # Test expectimax
    print("3. Testing expectimax (depth=2)...")
    best_move, best_score = choose_move(board, algorithm='expectimax', depth=2)
    if best_move:
        print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}\n")
    else:
        print("   ✗ expectimax not implemented")
        return
    
    # Test alpha-beta
    print("4. Testing alpha-beta (depth=3)...")
    import time
    start = time.time()
    best_move, best_score = choose_move(board, algorithm='alpha_beta', depth=3)
    elapsed = time.time() - start
    if best_move:
        print(f"   ✓ Best move: {best_move}, Score: {best_score:.2f}")
        print(f"   ✓ Time: {elapsed:.3f}s\n")
    else:
        print("   ✗ alpha_beta not implemented")
        return
    
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
    print(f"   Speedup: {minimax_time/ab_time:.2f}x\n")
    
    print("=== Exercise Complete ===")
    print("Compare these tree search methods with RL approaches!")


if __name__ == "__main__":
    main()

