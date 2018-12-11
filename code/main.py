import numpy as np
from utils import load_dataset
from utils import download_stock
from networks import StockAgentDQN,StockAgentDQNshort

# inputs=load_dataset(csv_path='./data/stock_data_FB.csv')

# SADQN=StockAgentDQN(input_data=inputs, epsilon = 0.05, test_week_num = 10, hard = 100, soft = 10, buy_action_weight = 1)

# for epoch in range(200):
# 	SADQN.learn()
# 	print('final_value=',SADQN.state[5]+SADQN.state[6])
# 	if epoch % 10 == 0:
# 		print(SADQN.action_percentage_history[-1])
		

# print(np.array(SADQN.state_list)[:, 5:7])





inputs=load_dataset(csv_path='./data/stock_data_apple_q.csv')

SADQNshort=StockAgentDQNshort(input_data=inputs, epsilon = 0.05, test_week_num = 10, hard = 100, soft = 10, buy_action_weight = 1,sample_size=6)

for epoch in range(800):
	SADQNshort.learn()	
	if epoch % 20 == 0:
		print('start_weeks=',SADQNshort.sample_start_weeks)
		print(SADQNshort.action_percentage_history[-1])
		print('final_value=',SADQNshort.state[5]+SADQNshort.state[6])


SADQNshort.test_model()
SADQNshort.plot_cost()
		

 


# cash = 0
# stock = 0
# for week in range(inputs.shape[0] - 1):
# 	cash -= 10
# 	stock += 10
# 	stock *= (1 + inputs[week + 1, 3])

# print(cash, stock)

# run test set

# SADQN.get_first_state(train=False)
# current_state = SADQN.state

# for week in range(SADQN.test_week_num - 1):
# 	current_action = SADQN.forward_propogate(current_state, week, train=False)
# 	current_state = SADQN.state_transition(current_state, current_action, week, train=False)
# 	print(current_state[5:7])



#SADQN.plot_cost()
#SADQN.plot_reward()



