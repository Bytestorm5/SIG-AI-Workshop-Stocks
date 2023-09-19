import os
import json
import pandas as pd

def has_symbol(symbol: str, source='data') -> bool:
    return os.path.exists(os.path.join(source, symbol + '.json'))

def list_symbols(source='data') -> list[str]:
    file_names = []

    # List files in the directory
    for filename in os.listdir(source):
        if filename.endswith(".json"):
            # Remove the file extension and folder path
            file_name = os.path.splitext(filename)[0]
            file_names.append(file_name)

    return file_names

def get_symbol(symbol: str, source='data') -> pd.DataFrame:
    sym_data = json.load(open(os.path.join(source, symbol + '.json'), 'r'))
    return pd.DataFrame(sym_data)
    
