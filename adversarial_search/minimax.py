import numpy as np
from shared.state_representation import discretize_state

# Define the possible actions for the agent
actions = ['buy', 'sell', 'hold']

# Helper function to compute total portfolio value based on cash, stock holdings, and stock price
def evaluate_portfolio(cash, holding, price):
    return cash + holding * price

# Minimax algorithm with alpha-beta pruning for adversarial stock trading
def minimax(df, depth, index, is_maximizing, cash, holding, alpha, beta):
    # Terminal condition: depth limit reached or data exhausted
    if index >= len(df) - 1 or depth == 0:
        return evaluate_portfolio(cash, holding, df.iloc[index]['Close'].item()), None

    price = df.iloc[index]['Close'].item()

    if is_maximizing:
        max_eval = float('-inf')
        best_action = None

        for action in actions:
            new_cash, new_holding = cash, holding

            # Simulate action result
            if action == 'buy' and cash >= price:
                new_cash -= price
                new_holding += 1
            elif action == 'sell' and holding > 0:
                new_cash += holding * price
                new_holding = 0

            # Recursive minimax call (switch roles to the adversary)
            eval, _ = minimax(df, depth - 1, index + 1, False, new_cash, new_holding, alpha, beta)

            # Maximize reward
            if eval > max_eval:
                max_eval = eval
                best_action = action

            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off

        return max_eval, best_action

    else:
        # Market as adversary: chooses a shift to minimize your gains
        min_eval = float('inf')

        for market_shift in [-0.02, 0, 0.02]:  # Simulated price changes: -2%, 0%, +2%
            simulated_price = price * (1 + market_shift)
            eval = evaluate_portfolio(cash, holding, simulated_price)

            if eval < min_eval:
                min_eval = eval

            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off

        return min_eval, None