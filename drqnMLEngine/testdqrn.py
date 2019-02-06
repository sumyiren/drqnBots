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
from DRQNseller import dqrnSeller
from DRQNbuyer import dqrnBuyer

# Cartpole 
# State  -> x, x_dot, theta, theta_dot
# Action -> force (+1, -1)

# Parameter setting 
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
nSellers = 3
max_steps = 150
teamSpirit = 0.5
# DRQN Parameters
step_size = 50

sess=tf.Session()   
#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph('../output/localtest16/model-3000.meta')
saver.restore(sess, '../output/localtest16/model-3000')

sBa = None
bB = []

graph = tf.get_default_graph()

for i in range(nSellers):
    j = i
    bB.append(dqrnBuyer('BuyerAgent'+str(j)))
    bB[i].x = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/x:0')
    bB[i].rnn_batch_size = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_batch_size:0')
    bB[i].rnn_step_size  = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_step_size:0')
    bB[i].output = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/op_to_restore:0')

sBa = dqrnSeller('SellerAgent', nSellers, step_size)
sBa.x = graph.get_tensor_by_name('SellerAgent'+'/x:0')
sBa.rnn_batch_size = graph.get_tensor_by_name('SellerAgent'+'/rnn_batch_size:0')
sBa.rnn_step_size  = graph.get_tensor_by_name('SellerAgent'+'/rnn_step_size:0')
sBa.output = graph.get_tensor_by_name('SellerAgent'+'/op_to_restore:0')


def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_buyer = np.stack(obs_buyer)
    return obs_seller, obs_buyer
    
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn) 
    
world = world(nSellers, max_steps, teamSpirit)
obs_seller, obs_buyer = resetWorld(world)
    
for i in range(nSellers):
    bB[i].observation = obs_buyer[i]
    bB[i].observation_set = []
    for j in range(step_size):
        bB[i].observation_set.append(bB[i].observation)
sBa.observation = obs_seller
sBa.observation_set = []
for j in range(step_size):
    sBa.observation_set.append(sBa.observation)
    
#initializer
actions_buyer = np.random.uniform(low=-1, high=1, size=(nSellers,))
actions_seller = np.random.uniform(low=-1, high=1, size=(nSellers,))

#sellers
Q_value = sBa.output.eval(session=sess, feed_dict={sBa.x: np.reshape(sBa.observation_set, [step_size, sBa.flatten_size]), sBa.rnn_step_size: step_size})[0]
for i in range(nSellers):
    actions_seller[i] = clamp(Q_value[i], -1, 1)
sBa.action = actions_seller

#buyers
for i in range(nSellers):
    Q_value = bB[i].output.eval(session=sess, feed_dict={bB[i].x: bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
    actions_buyer[i] = clamp(Q_value[0], -1, 1)
    bB[i].action = actions_buyer[i]

#print at the start
for i in range(nSellers):
    print('nSeller:'+str(i))
#        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
    print('SellerAsk = ' +str(obs_seller)+ 'BuyerAsk = ' + str(obs_buyer[i]))


step = 0
while step < max_steps:
    state = 'Testing'
#    for i in range(nSellers):
#        print('nSeller:'+str(i))
##        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
#        print('SellerAsk = ' +str(obs_seller[i])+ 'BuyerAsk = ' + str(obs_buyer[i]))
        
    #sellers
    Q_value = sBa.output.eval(session=sess, feed_dict={sBa.x: np.reshape(sBa.observation_set, [step_size, sBa.flatten_size]), sBa.rnn_step_size: step_size})[0]
    for i in range(nSellers):
        actions_seller[i] = clamp(Q_value[i], -1, 1)
    sBa.action = actions_seller
    
    #buyers
    for i in range(nSellers):
        Q_value = bB[i].output.eval(session=sess, feed_dict={bB[i].x: bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
        actions_buyer[i] = clamp(Q_value[0], -1, 1)
        bB[i].action = actions_buyer[i]

    
    obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = world.step(actions_seller, actions_buyer)  
            
    print('-------------------------------------')
    for i in range(nSellers):
        print('nSeller:'+str(i))
    #        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
        print('SellerAsk = ' +str(obs_seller)+ 'BuyerAsk = ' + str(obs_buyer[i]))
    
    for i in range(nSellers):
        bB[i].observation = obs_buyer_[i]
        bB[i].observation_set.append(bB[i].observation)
        if len(bB[i].observation_set) > step_size:
            del bB[i].observation_set[0]
    sBa.observation = obs_seller_
    sBa.observation_set.append(sBa.observation)
    if len(sBa.observation_set) > step_size:
        del sBa.observation_set[0]
    step = step+1
    
    obs_buyer = obs_buyer_
    obs_seller = obs_seller_

#print at the end
print('-------------------------------------')
for i in range(nSellers):
    print('nSeller:'+str(i))
#        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
    print('SellerAsk = ' +str(obs_seller)+ 'BuyerAsk = ' + str(obs_buyer[i]))
    
    
    