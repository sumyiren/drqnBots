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
nSellers = 1
max_steps = 150

# DRQN Parameters
step_size = 50

sess=tf.Session()   
#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph('../output/model-500.meta')
saver.restore(sess, '../output/model-500')

sBA = dqrnSeller('SellerAgent')
bB = []

graph = tf.get_default_graph()
sBA.x = graph.get_tensor_by_name("SellerAgent/x:0")
sBA.rnn_batch_size = graph.get_tensor_by_name("SellerAgent/rnn_batch_size:0")
sBA.rnn_step_size  = graph.get_tensor_by_name("SellerAgent/rnn_step_size:0")
sBA.output = graph.get_tensor_by_name("SellerAgent/op_to_restore:0")

for i in range(nSellers):
    j = i
    bB.append(dqrnBuyer('BuyerAgent'+str(j)))
    bB[i].x = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/x:0')
    bB[i].rnn_batch_size = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_batch_size:0')
    bB[i].rnn_step_size  = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_step_size:0')
    bB[i].output = graph.get_tensor_by_name('BuyerAgent'+str(j)+'/op_to_restore:0')


def flattenList(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_seller = flattenList(obs_seller)
    obs_buyer = np.stack(obs_buyer)
    return obs_seller, obs_buyer
    
    
    
world = world(nSellers, max_steps)
obs_seller, obs_buyer = resetWorld(world)
sBA.observation = obs_seller
sBA.observation_set = []
for i in range(step_size):
    sBA.observation_set.append(sBA.observation)
    
for i in range(nSellers):
    bB[i].observation = obs_buyer[i]
    bB[i].observation_set = []
    for j in range(step_size):
        bB[i].observation_set.append(bB[i].observation)
    
#sellers
Q_value = sBA.output.eval(session=sess, feed_dict={sBA.x: sBA.observation_set, sBA.rnn_step_size: step_size})[0]
sBA.action = []
actions_seller = [world.action_space.sample()]*nSellers
for i in range(nSellers):
    #                sBA.action[i] = np.argmax(Q_value[i:i+3])
    Q_value_pseudo = Q_value[i:i+3]
    sBA_pseudo_action = np.zeros([Num_action])
    sBA_pseudo_action[np.argmax(Q_value_pseudo)] = 1
    sBA.action.extend(sBA_pseudo_action)
    action_step = np.argmax(sBA_pseudo_action)
    actions_seller[i] = action_step   

#buyers
actions_buyer = [world.action_space.sample()]*nSellers
for i in range(nSellers):
    Q_value = bB[i].output.eval(session=sess, feed_dict={bB[i].x: bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
    bB[i].action = np.zeros([Num_action])
    bB[i].action[np.argmax(Q_value)] = 1
    action_step = np.argmax(bB[i].action)
    actions_buyer[i] = action_step


step = 0
while step < max_steps:
    state = 'Testing'
    for i in range(nSellers):
        print('nSeller:'+str(i))
        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))

    Q_value = sBA.output.eval(session=sess, feed_dict={sBA.x: sBA.observation_set, sBA.rnn_step_size: step_size})[0]
    for i in range(nSellers):
        #                sBA.action[i] = np.argmax(Q_value[i:i+3])
        Q_value_pseudo = Q_value[i:i+3]
        sBA_pseudo_action = np.zeros([Num_action])
        sBA_pseudo_action[np.argmax(Q_value_pseudo)] = 1
        sBA.action.extend(sBA_pseudo_action)
        action_step = np.argmax(sBA_pseudo_action)
        actions_seller[i] = action_step

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
    obs_seller_ = flattenList(obs_seller_)
    
    for i in range(nSellers):
        bB[i].observation = obs_buyer_[i]
        bB[i].observation_set.append(bB[i].observation)
        if len(bB[i].observation_set) > step_size:
            del bB[i].observation_set[0]
    
    sBA.observation = obs_seller_
    sBA.observation_set.append(sBA.observation)
    if len(sBA.observation_set) > step_size:
        del sBA.observation_set[0]
    step = step+1
    
    obs_buyer = obs_buyer_

    
    
    
    