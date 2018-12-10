from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



start_date = '2013-12-16'
end_date = '2018-12-31'
panel_data = data.DataReader('AAPL', 'iex', start_date, end_date)

# data
c=np.array(panel_data['close'])

# data
length_data = ((len(c) - 19) // 5) + 1

vol = np.full(length_data, None)
delta1 = np.full(length_data, None)
delta2 = np.full(length_data, None)
delta3 = np.full(length_data, None)
current_price = np.full(length_data, None)

# start at day 15, so we have 3 weeks worth of data to look at beforehand.
for label_day in range(15, len(c) - 4, 5):

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
data = {'Volatility': vol, 'Delta1' : delta1, 'Delta2' : delta2, 'Delta3' : delta3, 'Current_price' : current_price}
df = pd.DataFrame(data)
df.to_csv("./data/stock_data_apple_q.csv")


   
