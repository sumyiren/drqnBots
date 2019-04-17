#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:15:45 2019

@author: sumyiren
"""

import numpy as np 
import tensorflow as tf 
from worldTest import world
from DRQNseller import dqrnSeller
from DRQNbuyer import dqrnBuyer

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
max_steps = 150
teamSpirit = 0.5
# DRQN Parameters
step_size = 50

class testClass():
    
    def __init__(self):
        self.nSellers = 3
        self.sess=tf.Session()   
        self.saver = tf.train.import_meta_graph('../output/localtest24/model-3500.meta')
        self.saver.restore(self.sess, '../output/localtest24/model-3500')
        self.sB = []
        self.bB = []
        self.graph = tf.get_default_graph()
        self.world = world(self.nSellers, max_steps, teamSpirit)
        for i in range(self.nSellers):
            j = i
            self.bB.append(dqrnBuyer('BuyerAgent'+str(j)))
            self.bB[i].x = self.graph.get_tensor_by_name('BuyerAgent'+str(j)+'/x:0')
            self.bB[i].rnn_batch_size = self.graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_batch_size:0')
            self.bB[i].rnn_step_size  = self.graph.get_tensor_by_name('BuyerAgent'+str(j)+'/rnn_step_size:0')
            self.bB[i].output = self.graph.get_tensor_by_name('BuyerAgent'+str(j)+'/op_to_restore:0')
        
            k = i
            self.sB.append(dqrnSeller('SellerAgent'+str(k)))
            self.sB[i].x = self.graph.get_tensor_by_name('SellerAgent'+str(k)+'/x:0')
            self.sB[i].rnn_batch_size = self.graph.get_tensor_by_name('SellerAgent'+str(k)+'/rnn_batch_size:0')
            self.sB[i].rnn_step_size  = self.graph.get_tensor_by_name('SellerAgent'+str(k)+'/rnn_step_size:0')
            self.sB[i].output = self.graph.get_tensor_by_name('SellerAgent'+str(k)+'/op_to_restore:0')


    def resetWorld(self, world):
        obs_seller, obs_buyer = self.world.reset()
        obs_seller = np.stack(obs_seller)
        obs_buyer = np.stack(obs_buyer)
        return obs_seller, obs_buyer
    
    def restart(self):
        self.obs_seller, self.obs_buyer = self.resetWorld(world)
    
        for i in range(self.nSellers):
            self.bB[i].observation = self.obs_buyer[i]
            self.bB[i].observation_set = []
            for j in range(step_size):
                self.bB[i].observation_set.append(self.bB[i].observation)
            self.sB[i].observation = self.obs_seller[i]
            self.sB[i].observation_set = []
            for j in range(step_size):
                self.sB[i].observation_set.append(self.sB[i].observation)
        return self.obs_seller, self.obs_buyer 
#        #sellers
#        self.actions_seller = [world.action_space.sample()]*self.nSellers
#        for i in range(self.nSellers):
#            Q_value = self.sB[i].output.eval(session=self.sess, feed_dict={self.sB[i].x: self.sB[i].observation_set, self.sB[i].rnn_step_size: step_size})[0]
#            self.sB[i].action = np.zeros([Num_action])
#            self.sB[i].action[np.argmax(Q_value)] = 1
#            action_step = np.argmax(self.sB[i].action)
#            self.actions_seller[i] = action_step
#        
#        #buyers
#        self.actions_buyer = [world.action_space.sample()]*self.nSellers
#        for i in range(self.nSellers):
#            Q_value = self.bB[i].output.eval(session=self.sess, feed_dict={self.bB[i].x: self.bB[i].observation_set, bB[i].rnn_step_size: step_size})[0]
#            self.bB[i].action = np.zeros([Num_action])
#            self.bB[i].action[np.argmax(Q_value)] = 1
#            action_step = np.argmax(self.bB[i].action)
#            self.actions_buyer[i] = action_step
        
#        #print at the start
#        for i in range(self.nSellers):
#            print('nSeller:'+str(i))
#        #        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
#            print('SellerAsk = ' +str(obs_seller[i])+ 'BuyerAsk = ' + str(obs_buyer[i]))

    def stepAction(self):
        #sellers
        self.actions_seller = [self.world.action_space.sample()]*self.nSellers
        for i in range(self.nSellers):
            Q_value = self.sB[i].output.eval(session=self.sess, feed_dict={self.sB[i].x: self.sB[i].observation_set, self.sB[i].rnn_step_size: step_size})[0]
            self.sB[i].action = np.zeros([Num_action])
            self.sB[i].action[np.argmax(Q_value)] = 1
            action_step = np.argmax(self.sB[i].action)
            self.actions_seller[i] = action_step
        
        #buyers
        self.actions_buyer = [self.world.action_space.sample()]*self.nSellers
        for i in range(self.nSellers):
            Q_value = self.bB[i].output.eval(session=self.sess, feed_dict={self.bB[i].x: self.bB[i].observation_set, self.bB[i].rnn_step_size: step_size})[0]
            self.bB[i].action = np.zeros([Num_action])
            self.bB[i].action[np.argmax(Q_value)] = 1
            action_step = np.argmax(self.bB[i].action)
            self.actions_buyer[i] = action_step


        obs_seller_, obs_buyer_, rewards_seller, rewards_buyer, done \
            = self.world.step(self.actions_seller, self.actions_buyer)  
            
#        for i in range(self.nSellers):
#            print('nSeller:'+str(i))
#        #        print('SellerAsk = ' +str(obs_buyer[i][0])+ 'BuyerAsk = ' + str(obs_buyer[i][1]))
#            print('SellerAsk = ' +str(obs_seller[i])+ 'BuyerAsk = ' + str(obs_buyer[i]))
        
        for i in range(self.nSellers):
            self.bB[i].observation = obs_buyer_[i]
            self.bB[i].observation_set.append(self.bB[i].observation)
            if len(self.bB[i].observation_set) > step_size:
                del self.bB[i].observation_set[0]
            self.sB[i].observation = obs_seller_[i]
            self.sB[i].observation_set.append(self.sB[i].observation)
            if len(self.sB[i].observation_set) > step_size:
                del self.sB[i].observation_set[0]
        
        self.obs_buyer = obs_buyer_
        self.obs_seller = obs_seller_
        return obs_seller_, obs_buyer_, done
    
    
    
    