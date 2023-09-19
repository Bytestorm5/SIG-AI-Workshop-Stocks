import pandas as pd
import random

# TEMPLATE FILE
# ADD ANYTHING ELSE YOU LIKE, BUT YOU MUST HAVE THIS FUNCTION

# I reccomend making a separate file to actually train the model, and only having pre/post processing steps in here
# As well as next_price() obviously

def next_price(price_data: pd.DataFrame) -> float:
    # Should take the price data and return the next price of the stock
    return (random.random() * 2) * price_data['close'].iloc[-1]