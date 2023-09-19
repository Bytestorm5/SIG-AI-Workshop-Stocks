import pandas as pd

# This file just contains a bunch of functions to help do data transformation
# These are various indicators used in trading that are set up such that you can easily add them to your dataframe

# Every function will add a column to the dataframe as a side-effect

# If you want to make your own transform, don't edit this file since it won't be copied over when someone else uses it

def add_RSI(price_data: pd.DataFrame, period: int, basis: str='close'):
    """
    RSI is an indicator that returns a value between 0 and 100, indicating whether a stock is overpriced or underpriced.
    If the RSI is high, the stock is considered overpriced, and vice versa
    """
    price_diff = price_data[basis].diff(1)
    gains = price_diff.where(price_diff > 0, 0)
    losses = -price_diff.where(price_diff < 0, 0)

    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    price_data['RSI'] = rsi

def add_VWAP(price_data: pd.DataFrame):    
    """
    VWAP is the price of the stock weighted by volume.
    In essence, it helps find a value that's closer to the stock's "fair price" rather than simply the most recent price.
    """
    typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
    vwap = (typical_price * price_data['volume']).cumsum() / price_data['volume'].cumsum()
    
    price_data['VWAP'] = vwap

def add_BOLL(price_data: pd.DataFrame, window=20, std_dev=2, basis: str='close'):
    """
    Bollinger Bands show the average price over the specified period, as well as the price above and below the defined amount of standard deviations.
    In other words, it's a rolling confidence interval.
    """
    middle_band = price_data[basis].rolling(window=window).mean()
    std = price_data[basis].rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    price_data['Upper Bollinger Band'] = upper_band
    price_data['Middle Bollinger Band'] = middle_band
    price_data['Lower Bollinger Band'] = lower_band

def add_ZSCORE(price_data: pd.DataFrame, window=30, basis: str='close'):
    """
    The Z-Score is well, the Z-Score of the current price, assuming that the prices in the specified period are normally distributed
    (they're not)
    """
    middle_band = price_data[basis].rolling(window=window).mean()
    std = price_data[basis].rolling(window=window).std()

    price_data['Z-Score'] = (price_data[basis] - middle_band) / std

def add_MACD(price_data: pd.DataFrame, short_window=12, long_window=26, signal_window=9, basis: str='close'):
    """
    The MACD plots a "MACD" and "Signal" line, as well as a "Histogram" which is the diference between the two.
    A stock is considered to be doing well if the MACD is larger than the Signal, and vice versa. This can be easily measured with the histogram.
    """
    short_ema = price_data[basis].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = price_data[basis].ewm(span=long_window, min_periods=1, adjust=False).mean()
    
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    
    price_data['MACD'] = macd_line
    price_data['Signal'] = signal_line
    price_data['Histogram'] = macd_line - signal_line