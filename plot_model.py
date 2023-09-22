import kma_model as Model
import data_interface
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

symbol = "ORCL"

dataset: pd.DataFrame = data_interface.get_symbol(symbol, source='test_data')
size = len(dataset)

total_error = 0
total_datapoints = 0

real_prices = []
pred_prices = []

real_moves = []
pred_moves = []

for i in tqdm(list(range(2, size-2)), desc=symbol):
    in_data = dataset.head(i)
    out_price = dataset['close'].iloc[i+1]

    pred_price = Model.next_price(in_data)
    if pred_price == None:
        continue
    
    real_prices.append(out_price)
    pred_prices.append(pred_price)
    
    error = abs(out_price / pred_price)

    pred_move = (pred_price / in_data['close'].iloc[-1]) - 1
    real_move = (out_price / in_data['close'].iloc[-1]) - 1

    real_moves.append(real_move)
    pred_moves.append(pred_move)

    # Give more weight to cases in which the model moves the price in the wrong direction entirely
    if pred_move * real_move < 0:
        error *= 2

    total_error += error
    total_datapoints += 1
   
print(f"ERROR: {total_error / total_datapoints}\n - N = {total_datapoints}") 

plt.plot(real_prices)
plt.plot(pred_prices)

sim_prices = [real_prices[-1]]
for i in range(len(pred_moves)):
    sim_prices.append(sim_prices[-1] * (pred_moves[i]+1))
plt.plot(sim_prices)
plt.show()
