# pip install yfinance numpy pandas matplotlib
# python main.py

from data.fetch_data import get_stock_data
from hill_climbing.hill_climb import run_hill_climb
from reinforcement_learning.q_learning import train_q_agent
from reinforcement_learning.simulate_q import simulate_q_policy
from adversarial_search.simulate_adversary import simulate_adversarial_trading
from mdp.mdp_model import build_mdp
from mdp.policy_iteration import policy_iteration
from mdp.simulate_policy import simulate_policy
from utils.visualize import plot_portfolio

if __name__ == "__main__":
    print("Fetching stock data...")
    data = get_stock_data("AAPL", period="3mo")  # shorter period for fast test
    
    print("\n=== Test: Hill Climbing ===")
    run_hill_climb(data)
    
    print("\n=== Test: Q-Learning ===")
    train_q_agent(data, episodes=100)

    print("\n=== Test: Q-Learning Simulation ===")
    values = simulate_q_policy(data)
    plot_portfolio(values)

    print("\n=== Test: Adversarial Search ===")
    adv_values = simulate_adversarial_trading(data)

    print("\n=== Test: MDP ===")
    transitions, rewards = build_mdp(data)
    states = list(set(s for s, _ in transitions.keys()))
    policy = policy_iteration(states, ['buy', 'sell', 'hold'], transitions, rewards)
    mdp_values = simulate_policy(policy, data)