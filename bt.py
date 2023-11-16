import pandas_ta as ta
import os
import matplotlib.pyplot as plt
import seaborn as sns
import api_key as api
import ccxt
import sys
from datetime import datetime, timezone
import time
import pandas as pd
import requests
import numpy as np
import warnings
warnings.filterwarnings('ignore')

URL_BASE = r"https://api.glassnode.com"

###
model_params = [
    {
        'model_name': 'z_score',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 250, 10), 'l_threshold': np.arange(0, 2.5, 0.1)},
         {'resolution': '1h', 'l_window': np.arange(
             100, 3000, 100), 'l_threshold': np.arange(0, 2.5, 0.1)},
         {'resolution': '10m', 'l_window': np.arange(300, 10000, 300), 'l_threshold': np.arange(0, 2.5, 0.1)}]
    },
    {
        'model_name': 'ma_diff',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 200, 10), 'l_threshold': np.arange(0, 0.8, 0.02)},
         {'resolution': '1h', 'l_window': np.arange(
             100, 3000, 100), 'l_threshold': np.arange(0, 0.8, 0.02)},
         {'resolution': '10m', 'l_window': np.arange(300, 10000, 300), 'l_threshold': np.arange(0, 0.8, 0.02)}]
    },
    {
        'model_name': 'rsi',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 20, 1), 'l_threshold': np.arange(0, 30, 1)},
         {'resolution': '1h', 'l_window': np.arange(
             0, 20, 1), 'l_threshold': np.arange(0, 30, 1)},
         {'resolution': '10m', 'l_window': np.arange(0, 20, 1), 'l_threshold': np.arange(0, 30, 1)}]
    },
    {
        'model_name': 'percentile',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 20, 1), 'l_threshold': np.arange(0, 0.3, 0.02)},
         {'resolution': '1h', 'l_window': np.arange(
             0, 20*24, 1*24), 'l_threshold': np.arange(0, 0.3, 0.02)},
         {'resolution': '10m', 'l_window': np.arange(0, 20*24*6, 1*24*6), 'l_threshold': np.arange(0, 0.3, 0.02)}]
    },
    {
        'model_name': 'min_max',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(10, 110, 10), 'l_threshold': np.arange(0, 0.5, 0.05)},
         {'resolution': '1h', 'l_window': np.arange(
             10, 110, 10), 'l_threshold': np.arange(0, 0.5, 0.05)},
         {'resolution': '10m', 'l_window': np.arange(10, 110, 10), 'l_threshold': np.arange(0, 0.5, 0.05)}]
    },
    {
        'model_name': 'ma_cross',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 200, 10), 'l_threshold': np.arange(0, 200, 10)},
         {'resolution': '1h', 'l_window': np.arange(
             100, 3000, 100), 'l_threshold': np.arange(100, 3000, 100)},
         {'resolution': '10m', 'l_window': np.arange(300, 10000, 300), 'l_threshold': np.arange(300, 10000, 300)}]
    },
    {
        'model_name': 'iqr',
        'model_params':
        [{'resolution': '24h', 'l_window': np.arange(0, 200, 10), 'l_threshold': np.arange(0, 2.0, 0.02)},
         {'resolution': '1h', 'l_window': np.arange(
             100, 3000, 100), 'l_threshold': np.arange(0, 2.0, 0.02)},
         {'resolution': '10m', 'l_window': np.arange(300, 10000, 300), 'l_threshold': np.arange(0, 2.0, 0.02)}]
    },
]


def get_price_data(since, exchange: str = 'bybit', resolution: str = '24h', asset_output='BTC'):

    if exchange == 'bybit':
        exchange = ccxt.bybit()

    elif exchange == 'binance':
        exchange = ccxt.binance()

    else:
        sys.exit('Exchange not yet specify.')

    def fetch_ohlcv(symbol, timeframe, since):

        all_ohlcv = []
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if len(ohlcv) == 0:
                break
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        return pd.DataFrame(all_ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])

    if resolution == '10m':
        df = fetch_ohlcv(f"{asset_output}/USDT:USDT", '5m',
                         int(since.timestamp()*1000))  # <<<
        # df = df.drop(['o', 'h', 'l', 'v'], axis=1)
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('t').resample('10T').agg({'c': 'last'}).reset_index()
        df['t'] = df['t'].astype(int) // 10**9
    elif resolution == '24h':
        df = fetch_ohlcv(f"{asset_output}/USDT:USDT", '1d',
                         int(since.timestamp()*1000))  # <<<
        # df = df.drop(['o', 'h', 'l', 'v'], axis=1)
        df['t'] = df['t'] // 1000
        # df.columns=['t','v']
    elif resolution == '1h':
        df = fetch_ohlcv(f"{asset_output}/USDT:USDT", '1h',
                         int(since.timestamp()*1000))  # <<<
        # df = df.drop(['o', 'h', 'l', 'v'], axis=1)
        df['t'] = df['t'] // 1000
    else:
        sys.exit()

    # df['t'] = df['t'].astype(int).apply(datetime.fromtimestamp)
    # df.columns = ['t','v']
    return df


def get_df_endpoints(asset_input: str = 'BTC', resolution='24h'):

    res = requests.get(f"{URL_BASE}/v2/metrics/endpoints",
                       params={"api_key": api.api_glassnode['api_key']})
    df = pd.read_json(res.text)
    df = df[df['assets'].apply(lambda x: any(
        y['symbol'] == asset_input for y in x))]
    df = df[df['resolutions'].apply(lambda x: any(y == resolution for y in x))]

    return df


def get_df_value(endpoint, since, until, asset_input='BTC', resolution='24h'):

    res = requests.get(URL_BASE + endpoint,
                       params={
                           "a": asset_input,
                           "s": int(since.timestamp()),
                           "u": int(until.timestamp()),
                           "api_key": api.api_glassnode['api_key'],
                           "i": resolution,
                           "c": 'NATIVE',
                           "f": 'JSON'
                       }
                       )
    df_value = pd.read_json(res.text)
    # df_value['t'] = df_value['t'].astype(int).apply(datetime.fromtimestamp)
    return df_value


def backtesting(df_o, model, window, threshold, long_or_short, direction):

    df = df_o.copy()

    '''
    df is Dataframe merged with price and factor
    '''

    def z_score(df, window, threshold, long_or_short, direction):

        df['ma'] = df['factor'].rolling(window).mean()
        df['sd'] = df['factor'].rolling(window).std()
        # zscore is n standard d away from mean
        df['z'] = (df['factor'] - df['ma']) / df['sd']

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(df['z'] > threshold,
                                     1, np.where(df['z'] < -threshold, -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['z'] > threshold,
                                     1, np.where(df['z'] < -threshold, 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['z'] > threshold,
                                     0, np.where(df['z'] < -threshold, -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['z'] > threshold, -1, np.where(df['z'] < -threshold, 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['z'] > threshold,
                                     0, np.where(df['z'] < -threshold, 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['z'] > threshold, -1, np.where(df['z'] < -threshold, 0, 0))

        return df['pos']

    def ma_diff(df, window, threshold, long_or_short, direction):

        df['ma'] = df['factor'].rolling(window).mean()
        df['ma_diff'] = (df['factor'] / df['ma']) - 1.0
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(df['ma_diff'] > threshold, 1, np.where(
                    df['ma_diff'] < -threshold, -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['ma_diff'] > threshold, 1, np.where(
                    df['ma_diff'] < -threshold, 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['ma_diff'] > threshold, 0, np.where(
                    df['ma_diff'] < -threshold, -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['ma_diff'] > threshold, -1, np.where(df['ma_diff'] < -threshold, 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['ma_diff'] > threshold, 0, np.where(
                    df['ma_diff'] < -threshold, 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['ma_diff'] > threshold, -1, np.where(df['ma_diff'] < -threshold, 0, 0))

        return df['pos']

# models = ['z_score', 'ma_diff', 'rsi', 'ma_cross', 'iqr', 'min_max', 'percentile'][0:1]

    def rsi(df, window, threshold, long_or_short, direction):
        # print(df)

        df['rsi'] = ta.rsi(df['factor'], length=window)

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), 1, np.where(df['rsi'] < threshold, -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), 1, np.where(df['rsi'] < threshold, 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), 0, np.where(df['rsi'] < threshold, -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), -1, np.where(df['rsi'] < threshold, 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), 0, np.where(df['rsi'] < threshold, 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    (df['rsi'] > (100 - threshold)), -1, np.where(df['rsi'] < threshold, 0, 0))

        return df['pos']

    def ma_cross(df, window, threshold, long_or_short, direction):
        df['ma_1'] = df['factor'].rolling(window).mean()
        df['ma_2'] = df['factor'].rolling(threshold).mean()
        df['pre'] = np.sign(df['ma_1'].shift(1) - df['ma_2'].shift(1))
        df['post'] = np.sign(df['ma_1'] - df['ma_2'])

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where((df['pre'] < 0) & (df['post'] > 0), 1, np.where(
                    (df['pre'] > 0) & (df['post'] < 0), -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where((df['pre'] < 0) & (df['post'] > 0), 1, np.where(
                    (df['pre'] > 0) & (df['post'] < 0), 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where((df['pre'] < 0) & (df['post'] > 0), 0, np.where(
                    (df['pre'] > 0) & (df['post'] < 0), -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where((df['pre'] < 0) & (
                    df['post'] > 0), -1, np.where((df['pre'] > 0) & (df['post'] < 0), 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where((df['pre'] < 0) & (df['post'] > 0), 0, np.where(
                    (df['pre'] > 0) & (df['post'] < 0), 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where((df['pre'] < 0) & (
                    df['post'] > 0), -1, np.where((df['pre'] > 0) & (df['post'] < 0), 0, 0))
        return df['pos']

    def iqr(df, window, threshold, long_or_short, direction):

        df['q1'] = df['factor'].rolling(window).quantile(0.25)
        df['q3'] = df['factor'].rolling(window).quantile(0.75)
        df['iqr'] = df['q3'] - df['q1']
        df['lower_bound'] = df['q1'] - threshold * df['iqr']
        df['upper_bound'] = df['q3'] + threshold * df['iqr']

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], 1, np.where(
                    df['factor'] < df['lower_bound'], -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], 1, np.where(
                    df['factor'] < df['lower_bound'], 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], 0, np.where(
                    df['factor'] < df['lower_bound'], -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], -1,
                                     np.where(df['factor'] < df['lower_bound'], 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], 0, np.where(
                    df['factor'] < df['lower_bound'], 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['factor'] > df['upper_bound'], -1,
                                     np.where(df['factor'] < df['lower_bound'], 0, 0))
        return df['pos']

    def min_max(df, window, threshold, long_or_short, direction):

        df['min'] = df['factor'].rolling(window).min()
        df['max'] = df['factor'].rolling(window).max()
        df['x'] = (df['factor'] - df['min']) / (df['max'] - df['min'])

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), 1, np.where(df['x'] < (threshold), -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), 1, np.where(df['x'] < (threshold), 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), 0, np.where(df['x'] < (threshold), -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), -1, np.where(df['x'] < (threshold), 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), 0, np.where(df['x'] < (threshold), 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['x'] > (
                    1 - threshold), -1, np.where(df['x'] < (threshold), 0, 0))

        return df['pos']

    def percentile(df, window, threshold, long_or_short, direction):

        df['percentile_high'] = df['factor'].rolling(
            window).quantile(1 - threshold)
        df['percentile_low'] = df['factor'].rolling(window).quantile(threshold)

        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], 1, np.where(
                    df['factor'] < df['percentile_low'], -1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], 1, np.where(
                    df['factor'] < df['percentile_low'], 0, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], 0, np.where(
                    df['factor'] < df['percentile_low'], -1, 0))
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], -1,
                                     np.where(df['factor'] < df['percentile_low'], 1, 0))
            elif long_or_short == 'long':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], 0, np.where(
                    df['factor'] < df['percentile_low'], 1, 0))
            elif long_or_short == 'short':
                df['pos'] = np.where(df['factor'] > df['percentile_high'], -1,
                                     np.where(df['factor'] < df['percentile_low'], 0, 0))

        return df['pos']

    if model == 'z_score':
        df['pos'] = z_score(df, window, threshold, long_or_short, direction)
    if model == 'ma_diff':
        df['pos'] = ma_diff(df, window, threshold, long_or_short, direction)
    if model == 'rsi':
        df['pos'] = rsi(df, window, threshold, long_or_short, direction)
    if model == 'ma_cross':
        df['pos'] = ma_cross(df, window, threshold, long_or_short, direction)
    if model == 'iqr':
        df['pos'] = iqr(df, window, threshold, long_or_short, direction)
    if model == 'min_max':
        df['pos'] = min_max(df, window, threshold, long_or_short, direction)
    if model == 'percentile':
        df['pos'] = percentile(df, window, threshold, long_or_short, direction)

    df['pos_t-1'] = df['pos'].shift(1)
    df['trade'] = abs(df['pos_t-1'] - df['pos'])  # check有無交易
    df['cost'] = df['trade'] * 0.05/100  # 5bps
    df['pnl'] = df['pos_t-1'] * df['chg'] - df['cost']  # 扣咗交易費
    df['cumu'] = df['pnl'].cumsum()
    df['dd'] = df['cumu'].cummax() - df['cumu']

    df['bnh_pnl'] = df['chg']
    df.loc[0:(window-1), 'bnh_pnl'] = 0
    df['bnh_cumu'] = df['bnh_pnl'].cumsum()

    annual_return = round(df['pnl'].mean() * 365 * 1, 3)
    sharpe = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(365 * 1), 3)
    mdd = round(df['dd'].max(), 3)
    calmar = round(annual_return / mdd, 3)
    trades = df['trade'].sum()

    average_return = df.loc[window:len(df), 'pnl'].mean()
    sd = df.loc[window:len(df), 'pnl'].std()
    precise_sharpe = round(average_return / sd * np.sqrt(365 * 1), 3)

    # print(window, threshold, 'annual_return', annual_return, 'sharpe', sharpe, 'precise_sharpe', precise_sharpe, 'calmar', calmar, 'mdd', mdd, 'trades', trades)
    return pd.Series([window, threshold, precise_sharpe, calmar, mdd, trades], index=['window', 'threshold', 'sharpe', 'calmar', 'mdd', 'trades'])


### Asset data input & output ###

asset_input = 'BTC'
asset_output = 'BTC'

# Long_Short
long_or_short = ['long short', 'long', 'short']

# Direction
directions = ['momentum', 'reversion']

# Resolution - time frame
resolutions = ['10m', '1h', '24h'][2:]

# Model
models = ['z_score', 'ma_diff',  'ma_cross', 'iqr', 'min_max', 'percentile']
'rsi',
# Time frame
since = '2020-05-11'
until = '2023-10-31'
since = datetime.strptime(since, '%Y-%m-%d')
until = datetime.strptime(until, '%Y-%m-%d')

exchange = 'bybit'


def run(resolutions):

    for resolution in resolutions:
        df_end_pt = get_df_endpoints(asset_input, resolution)
        for index, row in df_end_pt[0:10].iterrows():
            df_factor_value = get_df_value(
                row['path'], since, until, asset_input, resolution)
            # get df price
            df_price = get_price_data(
                since, exchange, resolution, asset_output)
            df_merge = pd.merge(df_price, df_factor_value, 'inner', on='t')
            df_merge.columns = ['timestamp', 'open',
                                'high', 'low', 'close', 'volume', 'factor']
            df_merge['timestamp'] = df_merge['timestamp'].astype(int).apply(
                lambda x: datetime.fromtimestamp(x, timezone.utc))
            df_merge['chg'] = df_merge['close'].pct_change()
            # t = 1
            for model in models:
                for l_or_s in long_or_short:
                    for item in model_params:
                        if item['model_name'] == model:
                            for params in item['model_params']:
                                if params['resolution'] == '24h':
                                    l_window = params['l_window']
                                    l_threshold = params['l_threshold']
                                    # print(t)

                                    for direction in directions:
                                        # print(directions)
                                        # print(t,direction)
                                        results = []

                                        for threshold in l_threshold:
                                            for window in l_window:
                                                result = backtesting(
                                                    df_merge, model, window, threshold, l_or_s, direction)
                                                results.append(result)

                                        results = pd.DataFrame(results)
                                        results['threshold'] = results['threshold'].round(
                                            2)
                                        results['direction'] = direction
                                        results['long_or_short'] = l_or_s
                                        results['resolution'] = resolution
                                        results['model'] = model
                                        results['endpoint'] = row['path']

                                        csv_filename = f'results_{resolution}.csv'
                                        print(
                                            f'{index} | {row["path"]} | asset_input: {asset_input} | asset_output: {asset_output}| {model} | {l_or_s} | {params["resolution"]} | {direction}')
                                        if os.path.exists(csv_filename):
                                            o_data = pd.read_csv(csv_filename)
                                            o_data = pd.concat(
                                                [o_data, results])
                                            o_data.to_csv(
                                                csv_filename, index=False)

                                            # print('finish')

                                            del o_data
                                            del results

                                        else:
                                            results.to_csv(csv_filename)
                                            del results

                                        # t+=1


if __name__ == '__main__':

    run(resolutions)
