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

env = gym.make('CartPole-v0')
game_name = 'CartPole'
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
nSellers = 2
max_steps = 150

# DRQN Parameters
step_size = 149
lstm_size = 256
flatten_size = 4

sess=tf.Session()   
#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph('../output/model-15.meta')
saver.restore(sess, '../output/model-15')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name("SellerAgent/x:0")
rnn_batch_size = graph.get_tensor_by_name("SellerAgent/rnn_batch_size:0")
rnn_step_size  = graph.get_tensor_by_name("SellerAgent/rnn_step_size:0")
output = graph.get_tensor_by_name("SellerAgent/op_to_restore:0")

def flattenList(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

def resetWorld(world):
    obs_seller, obs_buyer = world.reset()
    obs_seller = np.stack(obs_seller)
    obs_seller = flattenList(obs_seller)
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

    
world = world(nSellers, max_steps)
obs_seller, obs_buyer = resetWorld(world)
observation = obs_seller
observation_set = []
for i in range(step_size):
    observation_set.append(observation)
    
Q_value = output.eval(session=sess, feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
action = []
actions_seller = [world.action_space.sample()]*nSellers
for i in range(nSellers):
    #                sBA.action[i] = np.argmax(Q_value[i:i+3])
    Q_value_pseudo = Q_value[i:i+3]
    sBA_pseudo_action = np.zeros([Num_action])
    sBA_pseudo_action[np.argmax(Q_value_pseudo)] = 1
    action.extend(sBA_pseudo_action)
    action_step = np.argmax(sBA_pseudo_action)
    actions_seller[i] = action_step   


step = 0
while step < max_steps:
    state = 'Testing'
    Q_value = output.eval(session=sess, feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
    actions_buyer = getUserBuyerActions(obs_buyer)
    for i in range(nSellers):
        #                sBA.action[i] = np.argmax(Q_value[i:i+3])
        Q_value_pseudo = Q_value[i:i+3]
        sBA_pseudo_action = np.zeros([Num_action])
        sBA_pseudo_action[np.argmax(Q_value_pseudo)] = 1
        action.extend(sBA_pseudo_action)
        action_step = np.argmax(sBA_pseudo_action)
        actions_seller[i] = action_step  
    
    obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = world.step(actions_seller, actions_buyer)   
    obs_seller_ = flattenList(obs_seller_)
    
    observation_next = obs_seller_
    observation = observation_next
    
    observation_set.append(observation)
    
    if len(observation_set) > step_size:
        del observation_set[0]
    step = step+1
    
    obs_buyer = obs_buyer_


#
#
## Initial parameters
#Replay_memory = []
#step = 1
#score = 0 
#episode = 0
#
#data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)
#
## DRQN variables
## Append episode data
#episode_memory = []
#observation_set = []
#
#observation = env.reset()
#action = env.action_space.sample()
#observation, reward, terminal, info = env.step(action)
#
## Figure and figure data setting
#plt.figure(1)
#plot_x = []
#plot_y = []
#
#observation_set.append(observation)
#observation_set.append(observation)
#observation_set.append(observation)
#observation_set.append(observation)
#observation_next = observation
#
## Making replay memory
#while True:
#    # Rendering
#    env.render()
#
#    if step > 4:
#        # Testing
#        state = 'Testing'
#        Q_value = output.eval(session=sess, feed_dict={x: observation_set, rnn_batch_size: 1, rnn_step_size: step_size})[0]
#
#        action = np.zeros([Num_action])
#        action[np.argmax(Q_value)] = 1
#        action_step = np.argmax(action)
#        
#        observation_next, reward, terminal, info = env.step(action_step)
#
#        Epsilon = 0
#
#
#    # Save experience to the Replay memory 
#    episode_memory.append([observation, action, reward, observation_next, terminal])
#
#    if len(Replay_memory) > Num_replay_memory:
#        del Replay_memory[0]
#
#    # Update parameters at every iteration    
#    step += 1
#    score += reward 
#    
#    if step%5000 == 0:
#        print('SAVED')
#        saver = tf.train.Saver()
#        saver.save(sess, './my_test_model',global_step=5000)
#
#    observation = observation_next
#
#    observation_set.append(observation)
#
#    if len(observation_set) > step_size:
#        del observation_set[0]
#
#    # Plot average score
#    if len(plot_x) % Num_episode_plot == 0 and len(plot_x) != 0 and state != 'Observing':
#        plt.xlabel('Episode')
#        plt.ylabel('Score')
#        plt.title('Cartpole_DRQN')
#        plt.grid(True)
#
#        plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
#        plt.draw()
#        plt.pause(0.000001)
#
#        plot_x = []
#        plot_y = [] 
#
#    # Terminal
#    if terminal == True:
#        print('step: ' + str(step) + ' / ' + 'episode: ' + str(episode) + ' / ' + 'state: ' + state  + ' / '  + 'epsilon: ' + str(Epsilon) + ' / '  + 'score: ' + str(score)) 
#
#        if len(episode_memory) > step_size:
#            Replay_memory.append(episode_memory)
#        episode_memory = []
#
#        # Plotting data
#        plot_x.append(episode)
#        plot_y.append(score)
#
#        score = 0
#        episode += 1
#        observation = env.reset()
#
#        observation_set = []
#        for i in range(step_size):
#            observation_set.append(observation)

