import matplotlib.pyplot as plt

# Plots the portfolio value over time
def plot_portfolio(values, label='Portfolio'):
    plt.plot(values, label=label)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.title("Trading Performance")
    plt.legend()
    plt.grid(True)
    plt.show()