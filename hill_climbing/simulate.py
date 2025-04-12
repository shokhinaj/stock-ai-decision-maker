# Simulates a trading strategy using a buy/sell threshold and calculates the final portfolio value
def simulate_strategy(df, threshold):
    cash = 1000
    holding = 0

    for i in range(len(df)):
        row = df.iloc[i]
        short_ma = row['Short_MA'].item()
        long_ma = row['Long_MA'].item()
        price = row['Close'].item()

        # Trading rule: Buy if short-term MA is above long-term MA by more than the threshold
        if short_ma > long_ma * (1 + threshold):
            if cash >= price:
                holding += 1
                cash -= price
        elif holding > 0:
            cash += holding * price
            holding = 0

    # Return final portfolio value
    return cash + holding * df.iloc[-1]['Close'].item()