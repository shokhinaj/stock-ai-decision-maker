# 📈 Stock Market Decision Maker — AI Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![yfinance](https://img.shields.io/badge/Data%20Source-YFinance-orange?logo=Yahoo&logoColor=white)](https://pypi.org/project/yfinance/)
[![AI Project](https://img.shields.io/badge/AI%20Course-Stock%20Decision%20Maker-success?logo=OpenAI&logoColor=white)](#)

---

### 💡 Project Overview

This project applies **four core AI techniques** to automate **stock market trading decisions**.  
The goal is to help an agent decide:  

> Should I **buy**, **sell**, or **hold** — based on historical stock data?

It is an educational experiment in AI decision-making, based on real market data, using the following approaches:

| AI Method               | Purpose                                   |
|--------------------------|-------------------------------------------|
| Hill Climbing            | Threshold-based search optimization       |
| Q-Learning               | Trial-and-error policy learning (RL)      |
| Adversarial Search       | Minimax strategy with market as opponent  |
| Markov Decision Process  | Probabilistic planning with policy iteration |

---

## 📊 Example Results

| Method               | Final Portfolio Value ($) |
|-----------------------|---------------------------|
| Hill Climbing         | $972.35                   |
| Q-Learning            | $852.24                   |
| Adversarial Search    | $1031.40                  |
| Markov Decision Process (MDP) | $1000.00          |

---

## 🧠 State Representation

All AI methods use the same 3-bit state:
(short_ma > long_ma, return > 0, volume_norm > 0)

This compact form enables consistent learning and decision-making across the models.

---

## ⚙️ How to Run

1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
2️⃣ Run the main script:
python main.py
3️⃣ Results will display portfolio values for:
	•	Hill Climbing
	•	Q-Learning (Training & Simulation)
	•	Adversarial Search
	•	Markov Decision Process (MDP)

Graphs of portfolio evolution will also pop up (if matplotlib is installed).

⸻

🎯 Project Purpose

This project connects real-world financial decision-making with:
	•	AI planning and search.
	•	Reinforcement learning.
	•	Adversarial reasoning.
	•	Probabilistic modeling.

You can see how each AI agent makes decisions under uncertainty and tries to maximize its returns over time.

⸻

📌 Future Improvements
	•	Add advanced stock indicators: RSI, MACD, Bollinger Bands.
	•	Integrate Deep Q-Learning (DQN) for more dynamic learning.
	•	Simulate portfolios across multiple assets.
	•	Use real-time streaming data for live AI trading experiments.

⸻

🧑‍💻 Authors

Emalyn Howard — Shokhina Jalilova — Christopher Meraz
CS 475/505: Artificial Intelligence I
Spring 2025 — Professor Christabel Wayllace

⸻

🚀 Educational Focus

This project demonstrates real-world applications of AI concepts:
	•	Search Algorithms (Hill Climbing).
	•	Reinforcement Learning (Q-Learning).
	•	Adversarial Game AI (Minimax).
	•	Markov Decision Processes (MDP).