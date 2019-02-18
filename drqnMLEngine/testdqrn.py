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
nSellers = 3
max_steps = 150
teamSpirit = 0.5
# DRQN Parameters
step_size = 50

sess=tf.Session()   
#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph('../output/localtest19/model-1500.meta')
saver.restore(sess, '../output/localtest19/model-1500')

sB = None
bB = []

graph = tf.get_default_graph()

for i in range(nSellers):
    j = i
    bB.append(dqrnBuyer('BuyerAgent'+str(j)))
    bB[i].x = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/x:0')
    bB[i].rnn_batch_size = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_batch_size:0')
    bB[i].rnn_step_size  = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_step_size:0')
    bB[i].output = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/op_to_restore:0')

sB = dqrnSeller('SellerAgent', nSellers)
sB.x = graph.get_tensor_by_name('SellerAgent'+'/x:0')
sB.rnn_batch_size = graph.get_tensor_by_name('SellerAgent'+'/rnn_batch_size:0')
sB.rnn_step_size  = graph.get_tensor_by_name('SellerAgent'+'/rnn_step_size:0')
sB.output = graph.get_tensor_by_name('SellerAgent'+'/op_to_restore:0')


def flattenList(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list


def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_buyer = np.stack(obs_buyer)
    return obs_seller, obs_buyer
    
    
world = world(nSellers, max_steps, teamSpirit)
obs_seller, obs_buyer = resetWorld(world)
    
for i in range(nSellers):
    bB[i].observation = obs_buyer[i]
    bB[i].observation_set = []
    for j in range(step_size):
        bB[i].observation_set.append(bB[i].observation)
sB.observation = obs_seller
sB.observation_set = []
for j in range(step_size):
    sB.observation_set.append(sB.observation)
    
#sellers
actions_seller = [world.action_space.sample()]*nSellers    
sB.action = np.zeros([Num_action*nSellers])
Q_value = sB.output.eval(session=sess, feed_dict={sB.x: sB.observation_set, sB.rnn_step_size: step_size})[0]
for i in range(nSellers):
    sB.action[i*3+np.argmax(Q_value[i*3:i*3+3])] = 1
    action_step = np.argmax(sB.action[i*3:i*3+3])
    actions_seller[i] = action_step
    
#buyers
actions_buyer = [world.action_space.sample()]*nSellers
for i in range(nSellers):
    Q_value = bB[i].output.eval(session=sess, feed_dict={bB[i].x: bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
    bB[i].action = np.zeros([Num_action])
    bB[i].action[np.argmax(Q_value)] = 1
    action_step = np.argmax(bB[i].action)
    actions_buyer[i] = action_step

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
    sB.action = np.zeros([Num_action*nSellers])
    actions_seller = [world.action_space.sample()]*nSellers    
    Q_value = sB.output.eval(session=sess, feed_dict={sB.x: sB.observation_set, sB.rnn_step_size: step_size})[0]
    for i in range(nSellers):
        sB.action[i*3+np.argmax(Q_value[i*3:i*3+3])] = 1
        action_step = np.argmax(sB.action[i*3:i*3+3])
        actions_seller[i] = action_step
    print(Q_value)
    print(sB.action)
    #buyers
    actions_buyer = [world.action_space.sample()]*nSellers
    for i in range(nSellers):
        Q_value = bB[i].output.eval(session=sess, feed_dict={bB[i].x: bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
        bB[i].action = np.zeros([Num_action])
        bB[i].action[np.argmax(Q_value)] = 1
        action_step = np.argmax(bB[i].action)
        actions_buyer[i] = action_step
    
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
    sB.observation = obs_seller_
    sB.observation_set.append(sB.observation)
    if len(sB.observation_set) > step_size:
        del sB.observation_set[0]
    step = step+1
    
    obs_buyer = obs_buyer_
    obs_seller = obs_seller_

#print at the end
print('-------------------------------------')
for i in range(nSellers):
    print('nSeller:'+str(i))
#        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
    print('SellerAsk = ' +str(obs_seller)+ 'BuyerAsk = ' + str(obs_buyer[i]))
    
    
    