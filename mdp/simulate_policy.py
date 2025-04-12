from shared.state_representation import discretize_state

# Simulates stock trading using the learned MDP policy
def simulate_policy(policy, df):
    cash = 1000
    holding = 0
    values = []

    for i in range(len(df)):
        row = df.iloc[i]
        state = discretize_state(row)  # Get current state from data
        price = row['Close'].item()

        # Use policy to decide the action for the current state
        action = policy.get(state, 'hold')

        if action == 'buy' and cash >= price:
            holding += 1
            cash -= price
        elif action == 'sell' and holding > 0:
            cash += holding * price
            holding = 0

        values.append(cash + holding * price)  # Log the portfolio value

    final = cash + holding * df.iloc[-1]['Close'].item()
    print(f"[MDP] Final Portfolio Value: ${final:.2f}")
    return values  # Return the list of portfolio values over time