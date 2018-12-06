from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[63]:


start_date = '2013-12-16'
end_date = '2018-12-31'
panel_data = data.DataReader('AAPL', 'iex', start_date, end_date)

# data
c=np.array(panel_data['close'])

# scores
scores = np.empty(len(c))

# labels
labels = np.empty(len(c))

for label_day in range(5, len(c) - 5, 5):
	# normalized standard deviation
	nstd = np.std(c[label_day - 1 : label_day + 4]) / np.mean(c[label_day - 1 : label_day + 4])

	# pecent change in stock price
	delta = (c[label_day + 4] - c[label_day - 1]) / c[label_day - 1]

	# ratio of these values
	ratio = delta / nstd

	# setting the score
	scores[label_day] = ratio

	# setting the label
	# -2 is hard sell, -1 soft sell, 0 do nothing, 1 soft buy, 2 hard buy

	# sort of arbirary cut-offs...
	if ratio < -3:
		label = -2
	elif ratio < -1:
		label = -1
	elif ratio < 1:
		label = 0
	elif ratio < 3:
		label = 1
	else:
		label = 2

	# assigning label
	labels[label_day] = label

print(scores)

data = {'Data': c, 'Scores': scores, 'Labels': labels}
df = pd.DataFrame(data)
df.to_csv("stock_data.csv")

   
