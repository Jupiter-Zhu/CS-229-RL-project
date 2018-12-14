import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import time



class StockAgentDQN:
    def __init__(
        self,       
        input_data,
        action_number=5, 
        state_num=7,       
        learning_rate=0.01,
        test_week_num=10,
        epsilon=0.05,
        hard=100,
        soft=10,
        buy_action_weight=1,
        gamma= 0.9                   
    ):

        self.test_week_num = test_week_num

        self.lr = learning_rate 

        self.n_y = action_number

        self.train_data = input_data[0:-test_week_num,:]

        self.num_train_weeks = self.train_data.shape[0] - 1
 
        self.test_data=input_data[-test_week_num:]   

        self.epsilon=epsilon

        self.gamma=gamma 

        self.n_x=state_num

        self.hard = hard

        self.soft = soft


        self.baw=buy_action_weight
        
        

        

        # Config for networks
        # n_l1 = 10
        # n_l2 = 10
        # W_init = tf.contrib.layers.xavier_initializer(seed=1)
        # b_init = tf.contrib.layers.xavier_initializer(seed=1)
        # self.build_network(n_l1, n_l2, W_init, b_init)
        

        self.sess = tf.Session()

        self.cost_history = []


        ####### q value in the game  ##############

        self.Q_target_list=[]

        self.Q_list=[]

        # # keeps track of current state
        #self.state=[]

        # stores list of states
        self.state_list=[]

        # stores list of rewards
        self.reward_list=[]

        # stores list of actions
        self.action_list=[]

        # 
        self.total_reward_history=[]

        self.action_percentage_history=[]

        
        

        self.sess.run(tf.global_variables_initializer())


    def get_first_state(self,train=True):

        # initialize state
        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        self.state = np.zeros(7)

        if train:
            self.state[0 : 5] = self.train_data[0, 0 : 5]
        else:
            self.state[0 : 5] = self.test_data[0, 0: 5]

        self.state[5] = 0
        self.state[6] = 0


        


    def state_transition(self,old_state, action, week, train=True):

        ################TODO_CAITLIN###############

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        # assuming our action are: -2 = hard sell, -1 = soft sell, 0 = no action, 1 = soft buy, 2 = hard buy

        # initializing new state
        new_state = np.zeros(len(old_state))

        # updating volatility, deltas, and current price to next week's values

        if train:
            new_state[ 0 : 5] = self.train_data[week + 1, 0 : 5]
        else:
            new_state[ 0 : 5] = self.test_data[week + 1, 0 : 5]

        # initialize current cash and stocks:
        new_state[5 : 7] = old_state[5 : 7]

        if action == -2:
            new_state[5] += self.hard
            new_state[6] -= self.hard
        elif action == -1:
            new_state[5] += self.soft
            new_state[6] -= self.soft
        elif action == 1:
            new_state[5] -= self.soft * self.baw
            new_state[6] += self.soft * self.baw
        elif action == 2:
            new_state[5] -= self.hard * self.baw
            new_state[6] += self.hard * self.baw

        # adusting stock valuation based on following week's price, using delta3 for the following week
        if train:
            new_state[6] *= (1 + self.train_data[week + 1, 3])
        else:
            new_state[6] *= (1 + self.test_data[week + 1, 3])


        ####smartsmartcode####

        return new_state


    def reward_function(self,state, action, week, train=True):
 

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        previous_value = state[5] + state[6]

        new_state = self.state_transition(state, action, week)

        new_value = new_state[5] + new_state[6]

        profit = new_value - previous_value

 
        return profit




    def store_state_actions_reward(self,s,a,r):

         


        self.state_list.append(s)
        self.action_list.append(a)
        self.reward_list.append(r)   
        
        




    def generate_target_q_list(self):       

        for week in range(self.num_train_weeks):
            current_state = self.state_list[week]

            current_action = self.action_list[week]

            current_reward = self.reward_list[week]

       
            state_prime = self.state_transition(old_state=current_state, action=current_action, week=week, train=True)

            state_prime_reshape=state_prime.reshape(7,1)
        
            max_Q_state_prime_a = np.max(self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_prime_reshape}))

            self.Q_target_list.append( current_reward + self.gamma * max_Q_state_prime_a )

         



    def forward_propogate(self,state,week,train=True):

        state_reshape=state.reshape(7,1)

        if train:     

        # If random sample from uniform distribution is less than the epsilon parameter then predict action, else take a random action
            if np.random.uniform() > self.epsilon:
            # Forward propagate to get q values of outputs



                actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

                q_value=np.max(actions_q_values)

            
                action = np.argmax(actions_q_values)-2

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state,a=action,r=reward )

                return action

                


            else:
            # Random action
                action = np.random.randint(0, 5)-2

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state, a=action, r=reward )

                return action

        else:
            actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

            action = np.argmax(actions_q_values)-2

            return action



    def learn(self):

        #reset lists

        self.state_list=[]

        
        self.reward_list=[]

        
        self.action_list=[]

        self.Q_target_list=[]

        self.get_first_state(train=True)

        for week in range(self.num_train_weeks):

            current_action = self.forward_propogate(self.state, week, train=True)

            self.state = self.state_transition(old_state=self.state,week=week,action=current_action,train=True)






        # Generate Q target values with Bellman equation
        self.generate_target_q_list()

        q_target_outputs=np.array(self.Q_target_list)

        batch_state=np.array(self.state_list).T

        # Train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={ self.X: batch_state, self.Y: q_target_outputs } )

        # Save cost
        self.cost_history.append(self.cost)

        self.total_reward_history.append(np.sum(np.array(self.reward_list)))

        action_num=np.zeros(5)

        counter=0

        for action in [-2,-1,0,1,2]:
            action_num[counter]=self.action_list.count(action)/len(self.action_list)
            counter+=1

        self.action_percentage_history.append(action_num)
        

    def build_network(self, n_l1, n_l2, W_init, b_init):
        ###########
        # EVAL NET
        ###########
        self.X = tf.placeholder(tf.float32, [self.n_x, None], name='s')
        self.Y = tf.placeholder(tf.float32, [ None ], name='Q_target')

        with tf.variable_scope('eval_net'):
            # Store variables in collection
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [n_l1, self.n_x], initializer=W_init, collections=c_names)
                 
                b1 = tf.get_variable('b1', [n_l1, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [n_l2, n_l1], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [n_l2, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.n_y, n_l2], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.n_y, 1], initializer=b_init, collections=c_names)

            # First layer
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X) + b1
                A1 = tf.nn.relu( Z1 )
            # Second layer
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu( Z2 )
            # Output layer
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_eval_outputs = Z3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, tf.reduce_max(self.q_eval_outputs, axis=0) ) )
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def test_model(self):
        self.get_first_state(train=False)
        for week in range(self.test_week_num - 1):
            current_action = self.forward_propogate(self.state, week, train=False)
            self.state = self.state_transition(self.state, current_action, week, train=False)
            print('current portofolio=(cash,stock)=',self.state[5:7])



    def plot_cost(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

    def plot_reward(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.total_reward_history)), self.total_reward_history)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()






###################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################

#only use short terms of training
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
###################################################################################################################




class StockAgentDQNshort:
    def __init__(
        self,       
        input_data,
        action_number=5, 
        state_num=7,       
        learning_rate=0.01,
        test_week_num=10,
        epsilon=0.05,
        hard=100,
        soft=10,
        buy_action_weight=1,
        gamma= 0.9,
        sample_size=4                   
    ):

        self.test_week_num = test_week_num

        self.lr = learning_rate 

        self.n_y = action_number

        self.train_data = input_data[0:-test_week_num,:]

        self.num_train_weeks = self.train_data.shape[0] - 1
 
        self.test_data=input_data[-test_week_num:]   

        self.epsilon=epsilon

        self.gamma=gamma 

        self.n_x=state_num

        self.hard = hard

        self.soft = soft

        self.sample_size=sample_size


        self.baw=buy_action_weight
        
        

        

        # Config for networks
        n_l1 = 10
        n_l2 = 10
        W_init = tf.contrib.layers.xavier_initializer(seed=1)
        b_init = tf.contrib.layers.xavier_initializer(seed=1)
        self.build_network(n_l1, n_l2, W_init, b_init)
        

        self.sess = tf.Session()

        self.cost_history = []


        ####### q value in the game  ##############

        self.Q_target_list=[]

        self.Q_list=[]

        # # keeps track of current state
        #self.state=[]

        # stores list of states
        self.state_list=[]

        # stores list of rewards
        self.reward_list=[]

        # stores list of actions
        self.action_list=[]

        # 
        self.total_reward_history=[]

        self.action_percentage_history=[]

        
        

        self.sess.run(tf.global_variables_initializer())


    def get_first_state(self, start_week=0, train=True):

        # initialize state
        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        self.state = np.zeros(7)

        if train:
            self.state[0 : 5] = self.train_data[start_week, 0 : 5]
        else:
            self.state[0 : 5] = self.test_data[0, 0: 5]

        self.state[5] = 0
        self.state[6] = 0


        


    def state_transition(self,old_state, action, week, train=True):

        ################TODO_CAITLIN###############

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        # assuming our action are: -2 = hard sell, -1 = soft sell, 0 = no action, 1 = soft buy, 2 = hard buy

        # initializing new state
        new_state = np.zeros(len(old_state))

        # updating volatility, deltas, and current price to next week's values

        if train:
            new_state[ 0 : 5] = self.train_data[week + 1, 0 : 5]
        else:
            new_state[ 0 : 5] = self.test_data[week + 1, 0 : 5]

        # initialize current cash and stocks:
        new_state[5 : 7] = old_state[5 : 7]

        if action == -2:
            new_state[5] += self.hard
            new_state[6] -= self.hard
        elif action == -1:
            new_state[5] += self.soft
            new_state[6] -= self.soft
        elif action == 1:
            new_state[5] -= self.soft * self.baw
            new_state[6] += self.soft * self.baw
        elif action == 2:
            new_state[5] -= self.hard * self.baw
            new_state[6] += self.hard * self.baw

        # adusting stock valuation based on following week's price, using delta3 for the following week
        if train:
            new_state[6] *= (1 + self.train_data[week + 1, 3])
        else:
            new_state[6] *= (1 + self.test_data[week + 1, 3])


        ####smartsmartcode####

        return new_state


    def reward_function(self,state, action, week, train=True):
 

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        previous_value = state[5] + state[6]

        new_state = self.state_transition(state, action, week)

        new_value = new_state[5] + new_state[6]

        profit = new_value - previous_value

 
        return profit




    def store_state_actions_reward(self,s,a,r):

         


        self.state_list.append(s)
        self.action_list.append(a)
        self.reward_list.append(r) 



    def get_sample_start_weeks(self):
        self.sample_start_weeks=np.random.randint(low=0,high=self.train_data.shape[0]-self.test_week_num-1,size=(self.sample_size,))
        
        




    def generate_target_q_list(self):
        for sample_num in range(self.sample_size):
            for week in range(self.test_week_num):

                current_state = self.state_list[ sample_num * self.test_week_num + week]

                current_action = self.action_list[sample_num * self.test_week_num + week]

                current_reward = self.reward_list[sample_num * self.test_week_num + week]

       
                state_prime = self.state_transition(old_state=current_state, action=current_action, week= self.sample_start_weeks[ sample_num ] + week, train=True)

                state_prime_reshape=state_prime.reshape(7,1)
        
                max_Q_state_prime_a = np.max(self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_prime_reshape}))

                self.Q_target_list.append( current_reward + self.gamma * max_Q_state_prime_a )

         



    def forward_propogate(self,state,week,train=True):

        state_reshape=state.reshape(7,1)

        if train:     

        # If random sample from uniform distribution is less than the epsilon parameter then predict action, else take a random action
            if np.random.uniform() > self.epsilon:
            # Forward propagate to get q values of outputs


                actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

                q_value=np.max(actions_q_values)

            
                action = np.argmax(actions_q_values)-2

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state,a=action,r=reward )

                return action

                


            else:
            # Random action
                action = np.random.randint(0, 5)-2

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state, a=action, r=reward )

                return action

        else:
            actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

            action = np.argmax(actions_q_values)-2
            
            return action



    def learn(self):

        #reset lists

        self.state_list=[]

        
        self.reward_list=[]

        
        self.action_list=[]

        self.Q_target_list=[]

        self.get_sample_start_weeks()

        for start_week in self.sample_start_weeks:
            self.get_first_state(start_week=start_week , train=True)
            for week in range(self.test_week_num):

                current_action = self.forward_propogate(self.state, start_week+week, train=True)

                self.state = self.state_transition(old_state=self.state,week=start_week+week,action=current_action,train=True)






        # Generate Q target values with Bellman equation
        self.generate_target_q_list()

        q_target_outputs=np.array(self.Q_target_list)

        batch_state=np.array(self.state_list).T

        # Train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={ self.X: batch_state, self.Y: q_target_outputs } )

        # Save cost
        self.cost_history.append(self.cost)

        self.total_reward_history.append(np.sum(np.array(self.reward_list)))

        action_num=np.zeros(5)

        counter=0

        for action in [-2,-1,0,1,2]:
            action_num[counter]=self.action_list.count(action)/len(self.action_list)
            counter+=1

        self.action_percentage_history.append(action_num)
        

    def build_network(self, n_l1, n_l2, W_init, b_init):
        ###########
        # EVAL NET
        ###########
        self.X = tf.placeholder(tf.float32, [self.n_x, None], name='s')
        self.Y = tf.placeholder(tf.float32, [ None ], name='Q_target')

        with tf.variable_scope('eval_net'):
            # Store variables in collection
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [n_l1, self.n_x], initializer=W_init, collections=c_names)
                b1 = tf.get_variable('b1', [n_l1, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [n_l2, n_l1], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [n_l2, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.n_y, n_l2], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.n_y, 1], initializer=b_init, collections=c_names)

            # First layer
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X) + b1
                A1 = tf.nn.relu( Z1 )
            # Second layer
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu( Z2 )
            # Output layer
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_eval_outputs = Z3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, tf.reduce_max(self.q_eval_outputs, axis=0) ) )
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def test_model(self):
        self.get_first_state(train=False)
        for week in range(self.test_week_num - 1):
            current_action = self.forward_propogate(self.state, week, train=False)
            self.state = self.state_transition(self.state, current_action, week, train=False)
            print('current portofolio=(cash,stock)=',self.state[5:7])



    def plot_cost(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

    def plot_reward(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.total_reward_history)), self.total_reward_history)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()






















################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################



#makes the reward relavant to the volatitlity



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


class StockAgentDQNbold:
    def __init__(
        self,       
        input_data,
        action_number=5, 
        state_num=7,       
        learning_rate=0.01,
        test_week_num=10,
        epsilon=0.05,
        hard=100,
        soft=10,
        buy_action_weight=1,
        gamma= 0.9,
        volatility_weight=1,
        exploration_hard_chance=0.2                   
    ):

        self.test_week_num = test_week_num

        self.lr = learning_rate 

        self.n_y = action_number

        self.train_data = input_data[0:-test_week_num,:]

        self.num_train_weeks = self.train_data.shape[0] - 1
 
        self.test_data=input_data[-test_week_num:]   

        self.epsilon=epsilon

        self.gamma=gamma 

        self.n_x=state_num

        self.hard = hard

        self.soft = soft

        self.volatility_weight=volatility_weight

        self.exploration_hard_chance=exploration_hard_chance


        self.baw=buy_action_weight
        
        

        

        # Config for networks
        # n_l1 = 10
        # n_l2 = 10
        # W_init = tf.contrib.layers.xavier_initializer(seed=1)
        # b_init = tf.contrib.layers.xavier_initializer(seed=1)
        # self.build_network(n_l1, n_l2, W_init, b_init)
        

        self.sess = tf.Session()

        self.cost_history = []


        ####### q value in the game  ##############

        self.Q_target_list=[]

        self.Q_list=[]

        # # keeps track of current state
        #self.state=[]

        # stores list of states
        self.state_list=[]

        # stores list of rewards
        self.reward_list=[]

        # stores list of actions
        self.action_list=[]

        # 
        self.total_reward_history=[]

        self.action_percentage_history=[]

        
        

        self.sess.run(tf.global_variables_initializer())


    def get_first_state(self,train=True):

        # initialize state
        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        self.state = np.zeros(7)

        if train:
            self.state[0 : 5] = self.train_data[0, 0 : 5]
        else:
            self.state[0 : 5] = self.test_data[0, 0: 5]

        self.state[5] = 0
        self.state[6] = 0


        


    def state_transition(self,old_state, action, week, train=True):

        ################TODO_CAITLIN###############

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        # assuming our action are: -2 = hard sell, -1 = soft sell, 0 = no action, 1 = soft buy, 2 = hard buy

        # initializing new state
        new_state = np.zeros(len(old_state))

        # updating volatility, deltas, and current price to next week's values

        if train:
            new_state[ 0 : 5] = self.train_data[week + 1, 0 : 5]
        else:
            new_state[ 0 : 5] = self.test_data[week + 1, 0 : 5]

        # initialize current cash and stocks:
        new_state[5 : 7] = old_state[5 : 7]

        if action == -2:
            new_state[5] += self.hard
            new_state[6] -= self.hard
        elif action == -1:
            new_state[5] += self.soft
            new_state[6] -= self.soft
        elif action == 1:
            new_state[5] -= self.soft * self.baw
            new_state[6] += self.soft * self.baw
        elif action == 2:
            new_state[5] -= self.hard * self.baw
            new_state[6] += self.hard * self.baw

        # adusting stock valuation based on following week's price, using delta3 for the following week
        if train:
            new_state[6] *= (1 + self.train_data[week + 1, 3])
        else:
            new_state[6] *= (1 + self.test_data[week + 1, 3])


        ####smartsmartcode####

        return new_state


    def reward_function(self,state, action, week, train=True):
 

        # assuming that state is a 7-vector of the form:
        # (volatility, delta1, delta2, delta3, current price, current cash, current amount ($) of stock owned)

        previous_value = state[5] + state[6]

        new_state = self.state_transition(state, action, week)

        new_value = new_state[5] + new_state[6]

        profit = new_value - previous_value

 
        return profit * (1 +  self.volatility_weight* state[0])




    def store_state_actions_reward(self,s,a,r):

         


        self.state_list.append(s)
        self.action_list.append(a)
        self.reward_list.append(r)   
        
        




    def generate_target_q_list(self):       

        for week in range(self.num_train_weeks):
            current_state = self.state_list[week]

            current_action = self.action_list[week]

            current_reward = self.reward_list[week]

       
            state_prime = self.state_transition(old_state=current_state, action=current_action, week=week, train=True)

            state_prime_reshape=state_prime.reshape(7,1)
        
            max_Q_state_prime_a = np.max(self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_prime_reshape}))

            self.Q_target_list.append( current_reward + self.gamma * max_Q_state_prime_a )

         



    def forward_propogate(self,state,week,train=True):

        state_reshape=state.reshape(7,1)

        if train:     

        # If random sample from uniform distribution is less than the epsilon parameter then predict action, else take a random action
            if np.random.uniform() > self.epsilon:
            # Forward propagate to get q values of outputs



                actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

                q_value=np.max(actions_q_values)

            
                action = np.argmax(actions_q_values)-2

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state,a=action,r=reward )

                return action

                


            else:
            # Random action
                other_prob= (1-2 * self.exploration_hard_chance )/3

                action_distribution=np.array([self.exploration_hard_chance, other_prob,other_prob,other_prob ,self.exploration_hard_chance])

                possible_actions=np.array([-2,-1,0,1,2])


                
                action=np.random.choice(possible_actions,p=action_distribution)


                

                reward=self.reward_function(state, action, week, train=True)


                self.store_state_actions_reward( s=state, a=action, r=reward )

                return action

        else:
            actions_q_values = self.sess.run(self.q_eval_outputs, feed_dict={self.X: state_reshape})

            action = np.argmax(actions_q_values)-2

            return action



    def learn(self):

        #reset lists

        self.state_list=[]

        
        self.reward_list=[]

        
        self.action_list=[]

        self.Q_target_list=[]

        self.get_first_state(train=True)

        for week in range(self.num_train_weeks):

            current_action = self.forward_propogate(self.state, week, train=True)

            self.state = self.state_transition(old_state=self.state,week=week,action=current_action,train=True)






        # Generate Q target values with Bellman equation
        self.generate_target_q_list()

        q_target_outputs=np.array(self.Q_target_list)

        batch_state=np.array(self.state_list).T

        # Train eval network
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={ self.X: batch_state, self.Y: q_target_outputs } )

        # Save cost
        self.cost_history.append(self.cost)

        self.total_reward_history.append(np.sum(np.array(self.reward_list)))

        action_num=np.zeros(5)

        counter=0

        for action in [-2,-1,0,1,2]:
            action_num[counter]=self.action_list.count(action)/len(self.action_list)
            counter+=1

        self.action_percentage_history.append(action_num)
        

    def build_network(self, n_l1, n_l2, W_init, b_init):
        ###########
        # EVAL NET
        ###########
        self.X = tf.placeholder(tf.float32, [self.n_x, None], name='s')
        self.Y = tf.placeholder(tf.float32, [ None ], name='Q_target')

        with tf.variable_scope('eval_net'):
            # Store variables in collection
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('parameters'):
                W1 = tf.get_variable('W1', [n_l1, self.n_x], initializer=W_init, collections=c_names)
                 
                b1 = tf.get_variable('b1', [n_l1, 1], initializer=b_init, collections=c_names)
                W2 = tf.get_variable('W2', [n_l2, n_l1], initializer=W_init, collections=c_names)
                b2 = tf.get_variable('b2', [n_l2, 1], initializer=b_init, collections=c_names)
                W3 = tf.get_variable('W3', [self.n_y, n_l2], initializer=W_init, collections=c_names)
                b3 = tf.get_variable('b3', [self.n_y, 1], initializer=b_init, collections=c_names)

            # First layer
            with tf.variable_scope('layer_1'):
                Z1 = tf.matmul(W1, self.X) + b1
                A1 = tf.nn.relu( Z1 )
            # Second layer
            with tf.variable_scope('layer_2'):
                Z2 = tf.matmul(W2, A1) + b2
                A2 = tf.nn.relu( Z2 )
            # Output layer
            with tf.variable_scope('layer_3'):
                Z3 = tf.matmul(W3, A2) + b3
                self.q_eval_outputs = Z3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, tf.reduce_max(self.q_eval_outputs, axis=0) ) )
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def test_model(self):
        self.get_first_state(train=False)
        for week in range(self.test_week_num - 1):
            current_action = self.forward_propogate(self.state, week, train=False)
            self.state = self.state_transition(self.state, current_action, week, train=False)
            print('current portofolio=(cash,stock)=',self.state[5:7])



    def plot_cost(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

    def plot_reward(self):
        import matplotlib
        #matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.total_reward_history)), self.total_reward_history)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()






###################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################
###################################################################################################################





