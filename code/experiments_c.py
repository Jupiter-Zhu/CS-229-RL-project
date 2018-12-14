import numpy as np
from utils import load_dataset
from utils import download_stock
import tensorflow as tf
from networks import StockAgentDQN,StockAgentDQNshort

inputs=load_dataset(csv_path='./data/stock_data_FB.csv')

SADQN=StockAgentDQN(input_data=inputs, epsilon = 0.01, test_week_num = 10, hard = 100, soft = 10, buy_action_weight = 1)
SADQN.build_network(n_l1 = 10,n_l2 = 10, W_init = tf.contrib.layers.xavier_initializer(seed=1), b_init = tf.contrib.layers.xavier_initializer(seed=1))

# running simulation:

SADQN.sess.run(tf.global_variables_initializer())



print('************************************************************ ')



# Training nn with 500 epochs
for epoch in range(500):
	SADQN.learn()
	
	if epoch % 40 == 0:
		print('portofolio value at epoch '+ str(epoch+1)+'=',SADQN.state[5]+SADQN.state[6])
		print('action percentage at  epoch '+ str(epoch+1)+'=',SADQN.action_percentage_history[-1])
	
print('Training Complete! :)\n ************************************************************ ')
print('Portfolio after training =: \n',np.array(SADQN.state_list)[:, 5] + np.array(SADQN.state_list)[:, 6])


# computing "only soft-buy" strategy result
cash = 0
stock = 0
for week in range(inputs.shape[0] - 1):
	cash -= 10
	stock += 10
	stock *= (1 + inputs[week + 1, 3])

print(cash + stock)






 