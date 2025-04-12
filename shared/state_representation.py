# Extracts a simplified state from stock market features
def discretize_state(row):
    return (
        (row['Short_MA'] > row['Long_MA']).item(),  # Whether short MA is greater than long MA
        (row['Return'] > 0).item(),                # Whether price return is positive
        (row['Volume_Norm'] > 0).item()            # Whether trading volume is above average
    )