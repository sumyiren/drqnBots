#!/usr/bin/env python5
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:18:58 2018

@author: sumyiren
"""

#Seller Environment


import gym
from gym.utils import seeding
class sellerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self, totalTime, sellerStartingPrice, buyerStartingPrice, minprice):
        self.sellerStartingPrice = sellerStartingPrice
        self.buyerStartingPrice = buyerStartingPrice
        self.minprice = minprice
        self.timeLeft = totalTime
#        self.starttime = time.time()

        # self.action_space = spaces.Discrete(5) #less, more, the same
#        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None, None, self.minprice, self.timeLeft, False
        

        self.steps_beyond_done = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_seller, action_buyer):
        state = self.state
        sellerask, buyerask, minprice, timeLeft = state
        if action_seller != None:
            plusminus_seller = self.AVAIL_TORQUE[action_seller]
            plusminus_buyer = self.AVAIL_TORQUE[action_buyer]
        else:
            plusminus_seller = 0

        sellerask += plusminus_seller
        buyerask += plusminus_buyer
        timeLeft -= 1
        done = timeLeft <= 0 
        done = bool(done)
        self.state = [sellerask, buyerask, minprice, timeLeft]
#        reward = self.calcReward(sellerask, buyerask, done) 
#        print("SellerBot")
#        print("sellerask: "+ str(sellerask))
#        print("buyerask: "+ str(buyerask))
        return self.state, done

    def calcReward(self, sellerask, buyerask , done):
        reward = 0
        
        
        if done:
            if abs(sellerask - buyerask) <= 1 :
                
#                if sellerask < self.minprice:
#                    reward += 0#-1 * abs(sellerask - self.minprice)
#                elif (sellerask >= self.minprice and sellerask < self.sellerStartingPrice):
#                    reward += abs(sellerask- self.minprice)
#                elif sellerask >= self.sellerStartingPrice:
#                    reward += 2* abs(sellerask - self.sellerStartingPrice)
                    
                if sellerask >= self.sellerStartingPrice:
                    reward += 2* abs(sellerask - self.sellerStartingPrice)
                else:
                    reward += 0.1* abs(sellerask - self.sellerStartingPrice)
                
            else:
                reward += -1 * abs(sellerask - buyerask)
                    
                    
            if sellerask <=0:
                reward += -2
                
        else:
            if sellerask < self.minprice:
                reward += -1
            if sellerask <=0:
                reward += -1

        return reward
        

    def reset(self):
        self.state = [self.sellerStartingPrice, self.buyerStartingPrice, self.minprice, self.timeLeft]
        return self.state

#    def close(self):
        
    
    
    
    
    
    