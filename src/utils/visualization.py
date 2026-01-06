"""
Visualization utilities for training analysis.

This module provides functions to plot learning curves and analyze training results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(episode_rewards, window=10, title="Learning Curve"):
    """
    Plot learning curve with moving average.
    
    Args:
        episode_rewards: List of rewards per episode
        window: Window size for moving average
        title: Plot title
    """
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Raw rewards
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode rewards')
    
    # Moving average
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                label=f'Moving average ({window} episodes)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print statistics
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Mean reward: {np.mean(episode_rewards):.3f}")
    print(f"Std reward: {np.std(episode_rewards):.3f}")
    print(f"Best episode: {np.max(episode_rewards):.3f}")
    print(f"Worst episode: {np.min(episode_rewards):.3f}")

