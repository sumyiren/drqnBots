# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Import modules 
import tensorflow as tf
import numpy as np

envs = []
algorithm = 'DRQN'

# Parameter setting 
nSellers = 3
Num_action = 3
Gamma = 0.99
Learning_rate = 0.00025

Num_start_training = 1000
Num_training = 2000
Num_testing  = 10000
Num_update = 250
Num_batch = 8
Num_episode_plot = 30

# DRQN Parameters
#step_size = 4
lstm_size = 256
flatten_size = 4*nSellers

class dqrnSeller(object):
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
        self.reward = 0
        self.terminal = False
        self.info = None
        self.step = 1
        self.score = 0
        self.episode = 0
        self.Num_batch = 6
        

        
    def build_model(self):

        with tf.variable_scope(self.scope):
            # Input
            self.x = tf.placeholder(tf.float32, shape = [None, 4*nSellers], name="x")
    
            self.w_fc = self.weight_variable([lstm_size, Num_action])
            self.b_fc = self.bias_variable([Num_action])
    
            self.rnn_batch_size = tf.placeholder(dtype = tf.int32, name="rnn_batch_size")
            self.rnn_step_size  = tf.placeholder(dtype = tf.int32, name="rnn_step_size")
    
            self.x_rnn = tf.reshape(self.x,[-1, self.rnn_step_size , flatten_size])
    
            with tf.variable_scope('network'):
                self.cell = tf.nn.rnn_cell.LSTMCell(num_units = lstm_size, state_is_tuple = True)
                self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(inputs = self.x_rnn, cell = self.cell, dtype = tf.float32)
    
            # Vectorization
            self.rnn_out = self.rnn_out[:, -1, :]
            self.rnn_out = tf.reshape(self.rnn_out, shape = [-1 , lstm_size])
    
            self.output = tf.add(tf.matmul(self.rnn_out, self.w_fc), self.b_fc, name="op_to_restore")
    
            # Loss function and Train
            self.action_target = tf.placeholder(tf.float32, shape = [None, Num_action], name="action_target")
            self.y_prediction = tf.placeholder(tf.float32, shape = [None], name="y_prediction")
    
            self.y_target = tf.reduce_sum(tf.multiply(self.output, self.action_target), reduction_indices = 1)
            self.Loss = tf.reduce_mean(tf.square(self.y_prediction - self.y_target))
            self.train_step = tf.train.AdamOptimizer(Learning_rate).minimize(self.Loss)

        # Initialize variables
    #        config = tf.ConfigProto(log_device_placement=True)
    #        config.gpu_options.allow_growth = True
    #        self.sess = tf.InteractiveSession(config=config)
    #        init = tf.global_variables_initializer()
    #        self.sess.run(init)



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

#    def saveModel(self, step, i):
#        saver = tf.train.Saver()
#        saver.save(self.sess, './seller_model_'+str(i),global_step=step)
        




        
        