# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules 
import tensorflow as tf 
import random
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
import time
import gym

envs = []
envs.append(gym.make('CartPole-v0'))
envs.append(gym.make('CartPole-v0'))
envs.append(gym.make('CartPole-v0'))
game_name = 'CartPole'
algorithm = 'DRQN'

# Parameter setting 
Num_action = 2
Gamma = 0.99
Learning_rate = 0.00025 
Epsilon = 1 
Final_epsilon = 0.01 

Num_replay_memory = 200
Num_start_training = 1000
Num_training = 2000
Num_testing  = 10000 
Num_update = 250
Num_batch = 8
Num_episode_plot = 30

# DRQN Parameters
step_size = 4
lstm_size = 256
flatten_size = 4

class dqrnAgent(object):
    def __init__(self, scope):
        self.scope = scope
        self.episode_memory = []
        self.observation_set = []
        self.Replay_memory = []
        self.minibatch = []
        self.batch_end_index = []
        self.count_minibatch = 0
        self.y_batch = []
        self.action_in = []
        self.observation = None
        self.observation_next = None
        self.action = None
        self.reward = None
        self.terminal = None
        self.info = None
        self.step = 1
        self.score = 0
        self.episode = 0
        self.epsilon = 1
        with tf.variable_scope(scope):
            # Build the graph
            self.build_model()

    def build_model(self):

        # Input
        self.x = tf.placeholder(tf.float32, shape = [None, 4], name="x")

        self.w_fc = self.weight_variable([lstm_size, Num_action])
        self.b_fc = self.bias_variable([Num_action])

        self.rnn_batch_size = tf.placeholder(dtype = tf.int32, name="rnn_batch_size")
        self.rnn_step_size  = tf.placeholder(dtype = tf.int32, name="rnn_step_size")

        self.x_rnn = tf.reshape(self.x,[-1, self.rnn_step_size , flatten_size])

        with tf.variable_scope('network'):
            self.cell = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size, state_is_tuple = True)
            self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.x_rnn, cell = self.cell, dtype = tf.float32)

        # Vectorization
        self.rnn_out = self.rnn_out[:, -1, :]
        self.rnn_out = tf.reshape(self.rnn_out, shape = [-1 , lstm_size])

        self.output = tf.add(tf.matmul(self.rnn_out, self.w_fc), self.b_fc, name="op_to_restore")

        # Loss function and Train
        self.action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
        self.y_prediction = tf.placeholder(tf.float32, shape = [None])

        self.y_target = tf.reduce_sum(tf.multiply(self.output, self.action_target), reduction_indices = 1)
        self.Loss = tf.reduce_mean(tf.square(self.y_prediction - self.y_target))
        self.train_step = tf.train.AdamOptimizer(Learning_rate).minimize(self.Loss)

        # Initialize variables
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)



    # Initialize weights and bias
    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)


    def get_output(self, obs, rnn_batch_size, rnn_step_size):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return self.output.eval(feed_dict={self.x: obs, self.rnn_batch_size: rnn_batch_size, self.rnn_step_size: rnn_step_size})[0]

    def get_output_batch(self, obs, rnn_batch_size, rnn_step_size):

        return self.output.eval(feed_dict={self.x: obs, self.rnn_batch_size: rnn_batch_size, self.rnn_step_size: rnn_step_size})


    def trainStep(self, action_in, y_batch, observation_batch, Num_batch, step_size):
        self.train_step.run(feed_dict = {self.action_target: action_in, self.y_prediction: y_batch, self.x: observation_batch, self.rnn_batch_size: Num_batch, self.rnn_step_size: step_size})

    def saveModel(self, step):
        saver = tf.train.Saver()
        saver.save(self.sess, './my_test_model',global_step=step)

#    def training


def dqrnLearning(envs):
    
    # Parameter setting 
    Num_action = 2
    Gamma = 0.99
    Learning_rate = 0.00025 
#    Epsilon = 1 
    Final_epsilon = 0.01 
    
    Num_replay_memory = 200
    Num_start_training = 1000
    Num_training = 2000
    Num_testing  = 10000 
    Num_update = 250
    Num_batch = 8
    Num_episode_plot = 30
    
    # DRQN Parameters
    step_size = 4
    lstm_size = 256
    flatten_size = 4
#    episode_memory = []
#    observation_set = []

    dA = []
    
    # Initial parameters
#    Replay_memory = []
#    step = 1
#    score = 0 
#    episode = 0
    for i in range(len(envs)):
        dA.append(dqrnAgent('Agent'+str(i)))
        dA[i].observation = envs[i].reset()
        dA[i].action = envs[i].action_space.sample()
        dA[i].observation, dA[i].reward, dA[i].terminal, dA[i].info = envs[i].step(dA[i].action)
        
    
        # Making replay memory
    while True:
        
        for i in range(len(envs)):
            # Rendering
            envs[i].render()
        
            if dA[i].step <= Num_start_training:
                state = 'Observing'
                dA[i].action = np.zeros([Num_action])
                dA[i].action[random.randint(0, Num_action - 1)] = 1.0
                action_step = np.argmax(dA[i].action)

                dA[i].observation_next, dA[i].reward, dA[i].terminal, dA[i].info = envs[i].step(action_step)
                dA[i].reward -= 5 * abs(dA[i].observation_next[0])

                if dA[i].step % 100 == 0:
                    print('step: ' + str(dA[i].step) + ' / '  + 'state: ' + state)
        
            elif dA[i].step <= Num_start_training + Num_training:
                # Training 
                state = 'Training'
        
                # if random value(0 - 1) is smaller than Epsilon, action is random. Otherwise, action is the one which has the largest Q value 
                if random.random() < Epsilon:
                    dA[i].action = np.zeros([Num_action])
                    dA[i].action[random.randint(0, Num_action - 1)] = 1.0
                    action_step = np.argmax(dA[i].action)
                        
                else:
                    Q_value = dA[i].get_output(dA[i].observation_set, 1, step_size)
    #                Q_value = output.eval(feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
                    dA[i].action = np.zeros([Num_action])
                    dA[i].action[np.argmax(Q_value)] = 1
                    action_step = np.argmax(dA[i].action)
                
                dA[i].observation_next,  dA[i].reward,  dA[i].terminal,  dA[i].info = envs[i].step(action_step)
                dA[i].reward -= 5 * abs(dA[i].observation_next[0])
        
                # Select minibatch
                episode_batch = random.sample(dA[i].Replay_memory, Num_batch)    
        
                dA[i].minibatch = []
                dA[i].batch_end_index = []
                dA[i].count_minibatch = 0
        
                for episode_ in episode_batch:
                    episode_start = np.random.randint(0, len(episode_) + 1 - step_size)
                    for step_ in range(step_size):
                        dA[i].minibatch.append(episode_[episode_start + step_])
                        if step_ == step_size - 1:
                            dA[i].batch_end_index.append(dA[i].count_minibatch)
        
                        dA[i].count_minibatch += 1
    
                # Save the each batch data 
                observation_batch      = [batch[0] for batch in dA[i].minibatch]
                action_batch           = [batch[1] for batch in dA[i].minibatch]
                reward_batch           = [batch[2] for batch in dA[i].minibatch]
                observation_next_batch = [batch[3] for batch in dA[i].minibatch]
                terminal_batch            = [batch[4] for batch in dA[i].minibatch]
        
                # # Update target network according to the Num_update value 
                # if step % Num_update == 0:
                #     assign_network_to_target()
        
                # Get y_prediction 
                dA[i].y_batch = []
                dA[i].action_in = []
                Q_batch = dA[i].get_output_batch(observation_next_batch, Num_batch, step_size)
    #            Q_batch = output.eval(feed_dict = {x: observation_next_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})
        
                for count, j in enumerate(dA[i].batch_end_index):
                    dA[i].action_in.append(action_batch[j])
                    if terminal_batch[j] == True:
                        dA[i].y_batch.append(reward_batch[j])
                    else:
                        dA[i].y_batch.append(reward_batch[j] + Gamma * np.max(Q_batch[count]))
                
                dA[i].trainStep(dA[i].action_in, dA[i].y_batch, observation_batch, Num_batch, step_size)
    #            train_step.run(feed_dict = {action_target: action_in, y_prediction: y_batch, x: observation_batch, rnn_batch_size: Num_batch, rnn_step_size: step_size})
        
                # Reduce epsilon at training mode 
                if dA[i].epsilon > Final_epsilon:
                    dA[i].epsilon -= 1.0/Num_training
                    
        
                
    #        elif step < Num_start_training + Num_training + Num_testing:
    #            # Testing
    #            state = 'Testing'
    #            Q_value = output.eval(feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
    #    
    #            action = np.zeros([Num_action])
    #            action[np.argmax(Q_value)] = 1
    #            action_step = np.argmax(action)
    #            
    #            observation_next, reward, terminal, info = env.step(action_step)
    #    
    #            Epsilon = 0
        
    #        else: 
    #            # Test is finished
    #            print('Test is finished!!')
    #            plt.savefig('./Plot/' + data_time + '_' + algorithm + '_' + game_name + '.png')    
    #            break
        
            # Save experience to the Replay memory 
            dA[i].episode_memory.append([dA[i].observation, dA[i].action, dA[i].reward, dA[i].observation_next, dA[i].terminal])
        
            if len(dA[i].Replay_memory) > Num_replay_memory:
                del dA[i].Replay_memory[0]
        
            # Update parameters at every iteration    
            dA[i].step += 1
            dA[i].score += dA[i].reward
            
            if dA[i].step%5000 == 0:
                print('SAVED')
                dA[i].saveModel(5000)
    
        
            dA[i].observation = dA[i].observation_next
        
            dA[i].observation_set.append(dA[i].observation)
        
            if len(dA[i].observation_set) > step_size:
                del dA[i].observation_set[0]
        
            # Terminal
            if dA[i].terminal == True:
                print('step: ' + str(dA[i].step) + ' / ' + 'episode: ' + str(dA[i].episode) + ' / ' + 'state: ' + state  + ' / '  + 'epsilon: ' + str(dA[i].epsilon) + ' / '  + 'score: ' + str(dA[i].score)) 
        
                if len(dA[i].episode_memory) > step_size:
                   dA[i].Replay_memory.append(dA[i].episode_memory)
                dA[i].episode_memory = []
        
                dA[i].score = 0
                dA[i].episode += 1
                dA[i].observation = envs[i].reset()
        
                dA[i].observation_set = []
                for j in range(step_size):
                    dA[i].observation_set.append(dA[i].observation)



dqrnLearning(envs)

