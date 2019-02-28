#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:52:18 2018

@author: sumyiren
"""
import tensorflow as tf 
import random
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
import time
import gym
from worldTest import world

# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

algorithm = 'DRQN'

# Parameter setting 
Num_action = 3
Gamma = 0.99
Learning_rate = 0.00025 
Epsilon = 1 
Final_epsilon = 0.01 

Num_replay_memory = 200
Num_start_training = 5000
Num_training = 25000
Num_testing  = 10000 
Num_update = 250
Num_batch = 8
Num_episode_plot = 30
nSellers = 1
max_steps = 150

# DRQN Parameters
step_size = 50
lstm_size = 256
flatten_size = 4
teamSpirit = 0.5
sess=tf.Session()   
#First let's load meta graph and restore weights


class testDrqn():
    
    def __init__(self):
        self.saver = tf.train.import_meta_graph('../output/localtest22_single/model-2000.meta')
        self.saver.restore(sess, '../output/localtest22_single/model-2000')
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("SellerAgent0/x:0")
        self.rnn_batch_size = self.graph.get_tensor_by_name("SellerAgent0/rnn_batch_size:0")
        self.rnn_step_size  = self.graph.get_tensor_by_name("SellerAgent0/rnn_step_size:0")
        self.output = self.graph.get_tensor_by_name("SellerAgent0/op_to_restore:0")
        
    def resetWorld(self, world):
        obs_seller, obs_buyer = world.reset()
        obs_seller = np.stack(obs_seller)
        obs_buyer = np.stack(obs_buyer)
        return obs_seller, obs_buyer
    
    
    def restart(self):
        self.world = world(nSellers, max_steps, teamSpirit)
        self.obs_seller, self.obs_buyer = self.resetWorld(self.world)
        observation = self.obs_seller[0]
        self.observation_set = []
        for i in range(step_size):
            self.observation_set.append(observation)
    
        Q_value = self.output.eval(session=sess, feed_dict={self.x: self.observation_set, self.rnn_batch_size: 1, self.rnn_step_size: step_size})[0]
        action = []
        self.actions_seller = [self.world.action_space.sample()]*nSellers

        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        action_step = np.argmax(action)
        self.actions_seller = action_step
        self.step = 0



    def stepAction(self, userAction):
        if self.step < max_steps:
            Q_value = self.output.eval(session=sess, feed_dict={self.x: self.observation_set, self.rnn_batch_size: 1, self.rnn_step_size: step_size})[0]
            actions_buyer =  [userAction]#getUserBuyerActions(obs_buyer)
            action = np.zeros([Num_action])
            action[np.argmax(Q_value)] = 1
            action_step = np.argmax(action)
            actions_seller = [action_step]
            
            obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
                    = self.world.step(actions_seller, actions_buyer)   
            
            observation_next = obs_seller_[0]
            observation = observation_next
            
            self.observation_set.append(observation)
            
            if len(self.observation_set) > step_size:
                del self.observation_set[0]
            self.step = self.step+1
            
            self.obs_buyer = obs_buyer_
            
            return self.obs_buyer
        
        else:
            return 'DONE'










