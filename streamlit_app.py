# streamlit_app.py

import streamlit as st
import pandas as pd
import io

from data.fetch_data import get_stock_data
from hill_climbing.hill_climb import run_hill_climb
from hill_climbing.simulate import simulate_strategy
from reinforcement_learning.q_learning import train_q_agent, get_q, actions as q_actions
from reinforcement_learning.simulate_q import simulate_q_policy
from adversarial_search.simulate_adversary import simulate_adversarial_trading
from adversarial_search.minimax import minimax
from mdp.mdp_model import build_mdp
from mdp.policy_iteration import policy_iteration
from mdp.simulate_policy import simulate_policy
from shared.state_representation import discretize_state

# Setup
st.set_page_config(page_title="AI Stock Trader", layout="centered")
st.title("📈 AI Stock Market Decision Maker")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo"])
method = st.radio("Choose an AI Strategy", ["Hill Climbing", "Q-Learning", "Adversarial Search", "MDP"])

run = st.button("▶️ Run Strategy")

if run:
    st.subheader(f"Fetching data for {ticker}...")
    df = get_stock_data(ticker, period)

    today = df.iloc[-1]
    today_state = discretize_state(today)
    today_price = today['Close'].item()

    # Initialize containers for outputs
    portfolio_values = []
    actions_list = []

    # --- Run Strategy ---
    if method == "Hill Climbing":
        st.subheader("🚀 Running Hill Climbing...")
        best_thresh, final_value = run_hill_climb(df)
        st.success(f"Final Portfolio Value: ${final_value:.2f}")

        # Simulate portfolio step-by-step
        cash, holding = 1000, 0
        for i in range(len(df)):
            row = df.iloc[i]
            short_ma = row['Short_MA'].item()
            long_ma = row['Long_MA'].item()
            price = row['Close'].item()

            if short_ma > long_ma * (1 + best_thresh):
                if cash >= price:
                    holding += 1
                    cash -= price
                action = "BUY"
            elif holding > 0:
                cash += holding * price
                holding = 0
                action = "SELL"
            else:
                action = "HOLD"

            portfolio_values.append(cash + holding * price)
            actions_list.append(action)

        st.line_chart(portfolio_values)

        # Today's Recommendation
        short_ma = today['Short_MA'].item()
        long_ma = today['Long_MA'].item()
        if short_ma > long_ma * (1 + best_thresh):
            today_action = "BUY"
            reason = f"Short MA ({short_ma:.2f}) > Long MA ({long_ma:.2f}) by threshold."
        else:
            today_action = "HOLD"
            reason = f"Short MA ({short_ma:.2f}) is not significantly greater than Long MA."

    elif method == "Q-Learning":
        st.subheader("🧠 Training Q-Learning agent...")
        train_q_agent(df, episodes=100)
        st.subheader("📊 Simulating Q-Learning policy...")
        
        cash, holding = 1000, 0
        for i in range(len(df)-1):
            row = df.iloc[i]
            state = discretize_state(row)
            price = row['Close'].item()

            q_vals = [get_q(state, a) for a in q_actions]
            action = q_actions[q_vals.index(max(q_vals))]

            if action == "buy" and cash >= price:
                holding += 1
                cash -= price
            elif action == "sell" and holding > 0:
                cash += holding * price
                holding = 0

            portfolio_values.append(cash + holding * price)
            actions_list.append(action.upper())

        st.success(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        st.line_chart(portfolio_values)

        # Today's Recommendation
        q_vals_today = [get_q(today_state, a) for a in q_actions]
        best_action = q_actions[q_vals_today.index(max(q_vals_today))]
        today_action = best_action.upper()
        reason = "Based on highest learned Q-value for today's state."

    elif method == "Adversarial Search":
        st.subheader("🤖 Running Adversarial Strategy...")

        cash, holding = 1000, 0
        depth = 2
        for i in range(len(df)-depth):
            price = df.iloc[i]['Close'].item()
            _, action = minimax(df, depth, i, True, cash, holding, float('-inf'), float('inf'))

            if action == "buy" and cash >= price:
                holding += 1
                cash -= price
            elif action == "sell" and holding > 0:
                cash += holding * price
                holding = 0

            portfolio_values.append(cash + holding * price)
            actions_list.append(action.upper() if action else "HOLD")

        st.success(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        st.line_chart(portfolio_values)

        # Today's Recommendation
        _, best_action = minimax(df, depth, len(df)-depth-1, True, cash=1000, holding=0, alpha=float('-inf'), beta=float('inf'))
        today_action = best_action.upper() if best_action else "HOLD"
        reason = "Action maximizing worst-case portfolio value."

    elif method == "MDP":
        st.subheader("📘 Running MDP Policy Iteration...")
        transitions, rewards = build_mdp(df)
        states = list(set(s for s, _ in transitions.keys()))
        policy = policy_iteration(states, ['buy', 'sell', 'hold'], transitions, rewards)
        
        cash, holding = 1000, 0
        for i in range(len(df)):
            row = df.iloc[i]
            state = discretize_state(row)
            price = row['Close'].item()

            action = policy.get(state, "hold")
            if action == "buy" and cash >= price:
                holding += 1
                cash -= price
            elif action == "sell" and holding > 0:
                cash += holding * price
                holding = 0

            portfolio_values.append(cash + holding * price)
            actions_list.append(action.upper())

        st.success(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        st.line_chart(portfolio_values)

        # Today's Recommendation
        today_action = policy.get(today_state, "hold").upper()
        reason = "Best policy action maximizing expected future rewards."

    # --- Display Today's Actionable Recommendation ---
    st.info(f"📈 Today's AI Recommendation ({method}): **{today_action}**")
    st.caption(f"Reason: {reason}")

    # --- Downloadable CSV including Actions ---
    portfolio_df = pd.DataFrame({
        "Step": list(range(len(portfolio_values))),
        "Portfolio Value": portfolio_values,
        "Action Taken": actions_list
    })

    csv_buffer = io.StringIO()
    portfolio_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Portfolio Data (with Actions)",
        data=csv_data,
        file_name=f"{ticker}_{method}_portfolio.csv",
        mime='text/csv'
    )