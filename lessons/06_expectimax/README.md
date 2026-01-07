# Lesson 6: Expectimax with Alpha-Beta Pruning

## Learning Objectives

By the end of this lesson, you will:
- Understand minimax algorithm for adversarial games
- Implement expectimax for stochastic opponents
- Apply alpha-beta pruning for efficiency
- Compare tree search methods with RL approaches
- Understand evaluation functions for chess

## Theory: Game Tree Search

### Minimax Algorithm

Minimax is a decision-making algorithm for adversarial games:
- **Maximizing player** (White) tries to maximize score
- **Minimizing player** (Black) tries to minimize score
- Recursively evaluates all possible moves to a certain depth
- Assumes optimal play from both players

**Algorithm:**
```
minimax(board, depth, maximizing_player):
    if depth == 0 or game_over:
        return evaluate(board)
    
    if maximizing_player:
        max_eval = -infinity
        for each move:
            eval = minimax(result(board, move), depth-1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = +infinity
        for each move:
            eval = minimax(result(board, move), depth-1, True)
            min_eval = min(min_eval, eval)
        return min_eval
```

### Expectimax

Expectimax handles stochastic (random) opponents:
- Instead of minimizing, takes **expected value** of opponent moves
- Useful when opponent is random or probabilistic
- More realistic for RL scenarios with exploration

**Algorithm:**
```
expectimax(board, depth, maximizing_player):
    if depth == 0 or game_over:
        return evaluate(board)
    
    if maximizing_player:
        return max(expectimax(result(board, move), depth-1, False) 
                   for each move)
    else:
        # Expectation over opponent moves
        moves = legal_moves(board)
        return sum(expectimax(result(board, move), depth-1, True) 
                   for move in moves) / len(moves)
```

### Alpha-Beta Pruning

Alpha-beta pruning optimizes minimax by eliminating branches:
- **Alpha**: Best value maximizer can guarantee
- **Beta**: Best value minimizer can guarantee
- Prune branches that can't improve the current best move
- Same result as minimax, but much faster!

**Key Insight:** If a move is worse than a previously examined move, we can skip it.

### Evaluation Function

A good evaluation function is crucial:
- **Material**: Piece values (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9)
- **Position**: Piece-square tables (center control, piece placement)
- **Mobility**: Number of legal moves
- **King Safety**: Is king under attack?
- **Pawn Structure**: Doubled, isolated, passed pawns

## Exercise Instructions

Open `exercise_expectimax.py` and implement:

1. **TODO 1**: Implement `evaluate_position(board)` - evaluation function
2. **TODO 2**: Implement `minimax(board, depth, maximizing_player)` - basic minimax
3. **TODO 3**: Implement `expectimax(board, depth, maximizing_player)` - stochastic version
4. **TODO 4**: Implement `alpha_beta(board, depth, alpha, beta, maximizing_player)` - optimized version
5. **TODO 5**: Implement `choose_move(board, algorithm='minimax', depth=3)` - move selection

**Learning Goals:**
- Understand how tree search works
- See the difference between minimax and expectimax
- Appreciate the speedup from alpha-beta pruning
- Compare with RL approaches (which learns instead of searching)

## Comparison with RL

**Tree Search (Minimax/Expectimax):**
- ✅ Optimal play (given perfect evaluation)
- ✅ No training needed
- ❌ Slow (exponential in depth)
- ❌ Requires good evaluation function

**Reinforcement Learning:**
- ✅ Learns from experience
- ✅ Can generalize to unseen positions
- ✅ Fast at inference time
- ❌ Requires training
- ❌ May not find optimal play

**Best of Both Worlds:** Use RL to learn evaluation function, then use minimax!

