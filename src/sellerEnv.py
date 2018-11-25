#!/usr/bin/env python5
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:18:58 2018

@author: sumyiren
"""

#Seller Environment


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
import torch as th
class sellerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self, totalTime, askingprice, minprice, buyeraskingprice):
        self.askingprice = askingprice
        self.minprice = minprice
        self.timeLeft = totalTime
        self.buyeraskingprice = buyeraskingprice
#        self.starttime = time.time()

        # self.action_space = spaces.Discrete(5) #less, more, the same
#        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None, None, self.timeLeft, False
        

        self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, buyerask_new):
        state = self.state
        sellerask, buyerask, timeLeft = state
        if action != None:
            plusminus = self.AVAIL_TORQUE[action]
        else:
            plusminus = 0

        sellerask += plusminus
        buyerask = buyerask_new
        timeLeft -= 1
        done = timeLeft <= 0 
        done = bool(done)
        self.state = [sellerask, buyerask, timeLeft]
#        reward = self.calcReward(sellerask, buyerask, done) 
#        print("SellerBot")
#        print("sellerask: "+ str(sellerask))
#        print("buyerask: "+ str(buyerask))
        return self.state, done

    def calcReward(self, sellerask, buyerask , done):
        reward = 0
        
        
        if done:
            if sellerask == buyerask:
                if sellerask < self.minprice:
                    reward += 2/(self.minprice - sellerask)#-(self.minprice-sellerask)/self.minprice*0.1
                elif sellerask >= self.askingprice:
                    reward += 2*(sellerask-self.askingprice)#(sellerask - self.askingprice)/self.askingprice*0.1
        #            print(reward)
                elif (sellerask >= self.minprice and sellerask < self.askingprice):
                    reward += 2
                    
            else:
                if sellerask < buyerask:
                    reward += -2 * abs(buyerask - sellerask)
                    
#                elif sellerask > buyerask:
#                    reward += -1*abs(sellerask-buyerask)
                    
            if sellerask <=0:
                reward += -2
                
        else:
            
#            if sellerask == buyerask:
#                reward += 1  
#            elif buyerask > sellerask:
#                reward += -1 * abs(buyerask - sellerask)                    
#            elif buyerask < sellerask:
#                reward += -1*abs(sellerask-buyerask)
                      
            if sellerask <=0:
                reward += -1
                

#        if sellerask == buyerask:
#            reward += 1
   
                

        return reward
        

    def reset(self):
        self.state = [self.askingprice, self.buyeraskingprice, self.timeLeft]
        return self.state

#    def close(self):
        
    
    
    
    
    
    