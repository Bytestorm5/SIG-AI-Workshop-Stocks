# Replace with model to test
import template as Model
import data_interface
import pandas as pd

def run_model(source='test_data') -> tuple[float, int]:
    total_error = 0
    total_datapoints = 0
    for symbol in data_interface.list_symbols(source=source):
        dataset: pd.DataFrame = data_interface.get_symbol(symbol, source=source)
        size = len(dataset)
        
        for i in range(2, size-2):
            in_data = dataset.head(i)
            out_price = dataset['close'].iloc[i+1]

            pred_price = Model.next_price(in_data)
            error = abs(out_price / pred_price)

            pred_move = (pred_price / in_data['close'].iloc[-1]) - 1
            real_move = (out_price / in_data['close'].iloc[-1]) - 1

            # Give more weight to cases in which the model moves the price in the wrong direction entirely
            if pred_move * real_move < 0:
                error *= 2

            total_error += error
            total_datapoints += 1
    
    return total_error / total_datapoints, total_datapoints

if __name__ == "__main__":
    train_error, count = run_model(source='data')
    print(f"TRAIN ERROR: {train_error}\n - N = {count}")

    test_error, count  = run_model(source='test_data')
    print(f"TEST ERROR:  {test_error}\n - N = {count}")