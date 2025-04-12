from shared.state_representation import discretize_state
from reinforcement_learning.q_learning import q_table, actions, get_q

# Simulates stock trading using a trained Q-learning policy
def simulate_q_policy(df):
    cash = 1000
    holding = 0
    portfolio_values = []

    for i in range(len(df) - 1):
        row = df.iloc[i]
        state = discretize_state(row)
        price = row['Close'].item()

        # Choose the best action from the learned Q-table
        q_vals = [get_q(state, a) for a in actions]
        action = actions[q_vals.index(max(q_vals))]

        if action == 'buy' and cash >= price:
            holding += 1
            cash -= price
        elif action == 'sell' and holding > 0:
            cash += holding * price
            holding = 0

        portfolio_value = cash + holding * price
        portfolio_values.append(portfolio_value)

    final_value = cash + holding * df.iloc[-1]['Close'].item()
    print(f"[Q-Simulation] Final Portfolio Value: ${final_value:.2f}")
    return portfolio_values