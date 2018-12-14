import numpy as np
from utils import load_dataset
from utils import download_stock
import tensorflow as tf
from networks import StockAgentDQN,StockAgentDQNshort, StockAgentDQNbold



###########################################################################

# gamma tuning experiement

inputs=load_dataset(csv_path='./data/stock_data_apple_q.csv')

SADQN=StockAgentDQN(input_data=inputs, epsilon = 0.05, test_week_num = 10, hard = 100, soft = 10, buy_action_weight = 1)
SADQN.build_network(n_l1 = 10,n_l2 = 10, W_init = tf.contrib.layers.xavier_initializer(seed=1), b_init = tf.contrib.layers.xavier_initializer(seed=1))

for gamma in [0.7]:


	SADQN.gamma=gamma

	SADQN.sess.run(tf.global_variables_initializer())




	print('************************************************************ ')

	 


	print('Training start with gamma value='+ str(gamma))

	

	for epoch in range(200):
		SADQN.learn()
		
		if epoch % 40 == 0:
			print('portofolio value at epoch '+ str(epoch+1)+'=',SADQN.state[5]+SADQN.state[6])
			print('action percentage at  epoch '+ str(epoch+1)+'=',SADQN.action_percentage_history[-1])
		
	print('Trainin Complete! :)\n ************************************************************ ')
	print(' portfolio value hitory after training with gamma= ' + str(gamma)+' =: \n',np.array(SADQN.state_list)[:, 5]+np.array(SADQN.state_list)[:, 6])

	SADQN.plot_cost()
	SADQN.plot_reward()
	 




#####################################################################################################
################ SADQNbold experiments
 
# inputs=load_dataset(csv_path='./data/stock_data_apple_q.csv')

# SADQN=StockAgentDQNbold(input_data=inputs, epsilon = 0.05, test_week_num = 10, hard = 100, soft = 10, buy_action_weight = 1,volatility_weight=1,exploration_hard_chance=0.2)
# SADQN.build_network(n_l1 = 10,n_l2 = 10, W_init = tf.contrib.layers.xavier_initializer(seed=1), b_init = tf.contrib.layers.xavier_initializer(seed=1))


# SADQN.sess.run(tf.global_variables_initializer())




# print('************************************************************ ')

	 


# print('Training start ')

	

# for epoch in range(500):
# 	SADQN.learn()
		
# 	if epoch % 40 == 0:
# 		print('portofolio value at epoch '+ str(epoch+1)+'=',SADQN.state[5]+SADQN.state[6])
# 		print('action percentage at  epoch '+ str(epoch+1)+'=',SADQN.action_percentage_history[-1])
		
# print('Trainin Complete! :)\n ************************************************************ ')
# print(' portfolio value hitory after training  =: \n',np.array(SADQN.state_list)[:, 5:7])

# SADQN.plot_cost()
# SADQN.plot_reward()
# 	 