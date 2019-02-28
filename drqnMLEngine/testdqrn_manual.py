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

saver = tf.train.import_meta_graph('../output/localtest22_single/model-2000.meta')
saver.restore(sess, '../output/localtest22_single/model-2000')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("SellerAgent0/x:0")
rnn_batch_size = graph.get_tensor_by_name("SellerAgent0/rnn_batch_size:0")
rnn_step_size  = graph.get_tensor_by_name("SellerAgent0/rnn_step_size:0")
output = graph.get_tensor_by_name("SellerAgent0/op_to_restore:0")


def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_buyer = np.stack(obs_buyer)
    return obs_seller, obs_buyer
    
    
def getUserBuyerActions(obs_buyer):
    actions_buyer = [world.action_space.sample()]*nSellers
    for i in range(nSellers):
        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
    for i in range(nSellers):
        val = input('Buyer '+str(i)+': ')
        actions_buyer[i] = int(val)
    return actions_buyer

    
world = world(nSellers, max_steps, teamSpirit)
obs_seller, obs_buyer = resetWorld(world)
observation = obs_seller[0]
observation_set = []
for i in range(step_size):
    observation_set.append(observation)
    
Q_value = output.eval(session=sess, feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
action = []
actions_seller = [world.action_space.sample()]*nSellers

action = np.zeros([Num_action])
action[np.argmax(Q_value)] = 1
action_step = np.argmax(action)
actions_seller = action_step


step = 0
while step < max_steps:
    state = 'Testing'
    Q_value = output.eval(session=sess, feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
    actions_buyer = getUserBuyerActions(obs_buyer)
    action = np.zeros([Num_action])
    action[np.argmax(Q_value)] = 1
    action_step = np.argmax(action)
    actions_seller = [action_step]
   

    
    obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = world.step(actions_seller, actions_buyer)   
    
    observation_next = obs_seller_[0]
    observation = observation_next
    
    observation_set.append(observation)
    
    if len(observation_set) > step_size:
        del observation_set[0]
    step = step+1
    
    obs_buyer = obs_buyer_



