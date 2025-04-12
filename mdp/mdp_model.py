from shared.state_representation import discretize_state

# Define the possible trading actions
actions = ['buy', 'sell', 'hold']

# Builds the Markov Decision Process model from stock data
def build_mdp(df):
    transitions = {}  # Dictionary mapping (state, action) -> list of possible next states
    rewards = {}      # Dictionary mapping (state, action) -> reward

    for i in range(len(df) - 1):
        state = discretize_state(df.iloc[i])        # Current state
        next_state = discretize_state(df.iloc[i + 1])  # Next day's state
        price = df.iloc[i]['Close'].item()
        next_price = df.iloc[i + 1]['Close'].item()

        for action in actions:
            reward = 0

            if action == 'buy':
                reward = -price  # Buying costs money
            elif action == 'sell':
                reward = price  # Selling earns money
            elif action == 'hold':
                reward = next_price - price  # Holding may gain or lose based on price change

            transitions.setdefault((state, action), []).append(next_state)
            rewards[(state, action)] = reward

    return transitions, rewards  # Return the transition and reward mappings for the MDP