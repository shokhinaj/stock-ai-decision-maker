from .simulate import simulate_strategy
import random

# Runs hill climbing optimization to find the best buy threshold for the strategy
def run_hill_climb(df, init_thresh=0.01, steps=100):
    best_thresh = init_thresh
    best_profit = simulate_strategy(df, best_thresh)

    for _ in range(steps):
        # Randomly tweak threshold slightly
        new_thresh = best_thresh + random.uniform(-0.005, 0.005)
        new_profit = simulate_strategy(df, new_thresh)

        # Keep the new threshold if profit improved
        if new_profit > best_profit:
            best_thresh, best_profit = new_thresh, new_profit

    print(f"[Hill Climb] Best Threshold: {best_thresh:.4f}, Profit: ${best_profit:.2f}")
    return best_thresh, best_profit