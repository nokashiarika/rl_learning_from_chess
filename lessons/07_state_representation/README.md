# Lesson 7: Multi-Feature State Representation with Matrix Operations

## Learning Objectives

By the end of this lesson, you will:
- Understand why single-feature states are limiting
- Implement multi-feature state encoding
- Use matrix operations for state representation
- Create board tensor representations (8x8x12)
- Apply convolutional and linear transformations
- See how better states improve RL learning

## Theory: State Representation in RL

### Why Single Features Fail

**Current approach:** Material count only (single float)
- ❌ Loses all positional information
- ❌ Many different positions map to same state
- ❌ Can't distinguish good vs bad piece placements
- ❌ No spatial understanding

**Example:** These positions have same material but very different:
- White Queen on center square (strong)
- White Queen on edge (weak)
- Both have material = 9, but positions are very different!

### Multi-Feature Approach

**Better approach:** Multiple features capturing different aspects
- ✅ Material count (what pieces exist)
- ✅ Piece positions (where pieces are)
- ✅ Positional values (piece-square tables)
- ✅ Tactical features (mobility, king safety)
- ✅ Strategic features (pawn structure, center control)

### Matrix Operations for State Encoding

#### 1. Board Tensor (8x8x12)

Represent board as 3D tensor:
- **8x8**: Chess board squares
- **12 channels**: One for each piece type/color
  - Channels: [White Pawn, White Knight, ..., Black King]

**Benefits:**
- Preserves spatial structure
- Can use **Convolutional Neural Networks (CNN)**
- Learns spatial patterns (pawn chains, piece coordination)

#### 2. Feature Vector

Extract hand-crafted features:
- Material balance
- Mobility (legal moves count)
- King safety
- Pawn structure
- Center control

**Then:** Use **matrix multiplication** to learn which features matter:
```
learned_features = W @ features + b
```
Where `W` is a learned weight matrix!

#### 3. Hybrid Approach

Combine both:
- Board tensor (spatial information)
- Feature vector (tactical/strategic information)
- Concatenate: `[flattened_tensor, features]`

### Matrix Operations Explained

#### Convolutional Layers (CNN)

```python
# Input: (8, 8, 12) board tensor
# Apply 3x3 convolution filters
# Learns: pawn chains, piece coordination, attack patterns
conv_output = conv2d(board_tensor, filters=32, kernel_size=3)
```

**What it learns:**
- Spatial patterns (e.g., "two pawns side-by-side")
- Local relationships (e.g., "knight near center")
- Attack formations (e.g., "queen and rook on same file")

#### Linear Transformations

```python
# Input: feature vector (n_features,)
# Learn which features are important
# Output: learned representation (hidden_dim,)
learned = linear(features, weight_matrix=W, bias=b)
```

**What it learns:**
- Feature importance (which features matter most)
- Feature interactions (how features combine)
- Dimensionality reduction (compress to essential info)

#### Attention Mechanisms

```python
# Learn which squares/pieces to focus on
attention_weights = softmax(Q @ K^T / sqrt(d)) @ V
```

**What it learns:**
- Which squares are critical (e.g., center, king area)
- Which pieces are important (e.g., active pieces)
- Dynamic focus (changes based on position)

## Exercise Instructions

Open `exercise_state_representation.py` and implement:

1. **TODO 1**: `board_to_tensor(board)` - Convert board to 8x8x12 tensor
2. **TODO 2**: `extract_features(board)` - Extract multi-feature vector
3. **TODO 3**: `encode_state(board, method='hybrid')` - Main encoding function
4. **TODO 4**: `piece_square_tables(board)` - Positional bonuses
5. **TODO 5**: `mobility_and_safety(board)` - Tactical features

**Learning Goals:**
- Understand multi-feature encoding
- See how matrix operations help
- Prepare states for neural network input
- Compare with single-feature approach

## Integration with RL

After encoding states:
1. Update environment `get_state()` to use new encoding
2. Update Gym wrapper `observation_space` shape
3. Train DQN with better states
4. Compare performance: single-feature vs multi-feature

**Expected improvement:**
- Better learning (more informative states)
- Higher win rates (can distinguish good/bad positions)
- Faster convergence (clearer signal)

