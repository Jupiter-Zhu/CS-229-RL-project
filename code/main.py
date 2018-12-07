import numpy as np
from utils import load_dataset
from networks import StockAgentDQN

inputs=load_dataset(csv_path='./data/stock_data_apple_q.csv')

SADQN=StockAgentDQN(input_data=inputs)

for epoch in range(100):
	SADQN.learn()
	print('final_value=',SADQN.state[5]+SADQN.state[6])



SADQN.plot_cost()



