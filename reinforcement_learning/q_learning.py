import random
from shared.state_representation import discretize_state

# Possible actions
actions = ['buy', 'sell', 'hold']

# Q-table for storing learned state-action values
q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration probability

# Get Q-value for a state-action pair
def get_q(state, action):
    return q_table.get((state, action), 0.0)

# Train a Q-learning agent on stock data
def train_q_agent(df, episodes=10):
    for _ in range(episodes):
        cash, holding = 1000, 0  # Initial conditions

        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            state = discretize_state(row)
            next_state = discretize_state(next_row)

            # Epsilon-greedy strategy: explore or exploit
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [get_q(state, a) for a in actions]
                action = actions[q_vals.index(max(q_vals))]

            price = row['Close'].item()
            next_price = next_row['Close'].item()
            reward = 0

            if action == 'buy' and cash >= price:
                holding += 1
                cash -= price
            elif action == 'sell' and holding > 0:
                cash += holding * price
                reward = (next_price - price) * holding
                holding = 0
            elif action == 'hold':
                reward = (next_price - price) * holding

            # Q-learning update rule
            max_q_next = max(get_q(next_state, a) for a in actions)
            q_table[(state, action)] = get_q(state, action) + alpha * (
                reward + gamma * max_q_next - get_q(state, action)
            )

    print("[Q-Learning] Training complete. Sample Q-values:")
    for k, v in list(q_table.items())[:5]:
        print(f"{k}: {v:.4f}")