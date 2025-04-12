from adversarial_search.minimax import minimax
import matplotlib.pyplot as plt

# Runs adversarial search-based trading strategy over the dataset
def simulate_adversarial_trading(df, depth=2):
    cash = 1000
    holding = 0
    portfolio_values = []

    for i in range(len(df) - depth):
        price = df.iloc[i]['Close'].item()

        # Use minimax to select the best action
        _, action = minimax(df, depth, i, True, cash, holding, float('-inf'), float('inf'))

        # Execute the chosen action
        if action == 'buy' and cash >= price:
            holding += 1
            cash -= price
        elif action == 'sell' and holding > 0:
            cash += holding * price
            holding = 0

        # Log the portfolio value at this step
        portfolio_values.append(cash + holding * price)

    # Calculate the final portfolio value
    final_value = cash + holding * df.iloc[-1]['Close'].item()
    print(f"[Adversarial Search] Final Portfolio Value: ${final_value:.2f}")
    return portfolio_values