import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd


def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    
    # Load headers
    # with open(csv_path, 'r') as csv_fh:
    #     headers = csv_fh.readline().strip().split(',')
    # print(headers)

    # Load features and labels
    # x_cols = [i for i in range(len(headers)) if headers[i].startswith('Data')]
    # print(x_cols)
    # l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    # print(l_cols)
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
     
    

    return inputs


def download_stock(ticker):
    """Download stock data to a CSV

    Args:
         ticker = ticker symbol for desired stock

    Returns:
        Nothing.  Creates a csv with the data.
    """

    start_date = '2013-12-16'
    end_date = '2018-12-31'
    panel_data = data.DataReader(ticker, 'iex', start_date, end_date)

    # data
    c=np.array(panel_data['close'])

    # data
    length_data = ((len(c) - 1) // 5) - 2

    vol = np.full(length_data, None)
    delta1 = np.full(length_data, None)
    delta2 = np.full(length_data, None)
    delta3 = np.full(length_data, None)
    current_price = np.full(length_data, None)

    # start at day 15, so we have 3 weeks worth of data to look at beforehand.
    for label_day in range(15, len(c), 5):

        # which row entry in our dataset
        week = (label_day - 15) // 5

        # volatility (normalized standard deviation) for three week period:
        vol[week] = np.std(c[label_day - 15 : label_day]) / np.mean(c[label_day - 15 : label_day])

        # pecent change in stock price in each week
        delta1[week] = (c[label_day - 11] - c[label_day - 15]) / c[label_day - 15]
        delta2[week] = (c[label_day - 6] - c[label_day - 10]) / c[label_day - 10]
        delta3[week] = (c[label_day - 1] - c[label_day - 5]) / c[label_day - 5]

        current_price[week] = c[label_day - 1]

    # write to csv
    info = {'Volatility': vol, 'Delta1' : delta1, 'Delta2' : delta2, 'Delta3' : delta3, 'Current_price' : current_price}
    df = pd.DataFrame(info)
    df.to_csv("./data/stock_data_" + ticker + ".csv")






