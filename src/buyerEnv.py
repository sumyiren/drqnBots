#!/usr/bin/env python5
# -*- coding: utf-8 -*-

#Seller Environment


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time

class buyerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self, totalTime, buyeraskingprice,askingprice, maxprice, determination):
        self.buyeraskingprice = buyeraskingprice
        self.askingprice = askingprice
        self.maxprice = maxprice
        self.timeLeft = totalTime
        self.determination = determination
#        self.starttime = time.time()

#        self.action_space = spaces.Discrete(5) #less, more, the same
#        self.observation_space = spaces.Box(-high, high)

        self.viewer = None
        self.state = None, None, self.timeLeft, self.determination, False
        

        self.steps_beyond_done = None


        
    def step(self, action, sellerask_new):
        state = self.state
        sellerask, buyerask, timeLeft, determination = state
        if action != None:
            plusminus = self.AVAIL_TORQUE[action]
        else:
            plusminus = 0
            
        buyerask += plusminus
        sellerask = sellerask_new
        timeLeft -= 1
        done = timeLeft <= 0 
        done = bool(done)
        self.state = (sellerask, buyerask, timeLeft, determination)       
#        print("BuyerBot")
#        print("sellerask: "+ str(sellerask))
#        print("buyerask: "+ str(buyerask))

        return np.array(self.state), done

    def calcReward(self, sellerask, buyerask , done):
        
        reward = 0
#        det_factor = 4**self.determination
        
        if done:
            if buyerask == sellerask:
                
                if buyerask > self.maxprice:
#                    reward += -2*(buyerask - self.maxprice)#-(buyerask-self.maxprice)/self.maxprice*0.1
                    reward += 2/(buyerask-self.maxprice)  
                    reward = reward*self.determination
                elif buyerask <= self.askingprice:
                    if buyerask > 0:
                        reward += 2*(self.askingprice - buyerask)#(self.askingprice - buyerask)/self.askingprice*0.1
                elif (buyerask <= self.maxprice and buyerask > self.askingprice):
                    reward += 2
                    reward += reward*self.determination
                    
#                reward += 4**self.determination
            else:
                if buyerask < sellerask:
                    reward += -2 * abs(buyerask - sellerask)

#                if buyerask > self.maxprice:
#                    reward += -0.5*(buyerask - self.maxprice)#-(buyerask-self.maxprice)/self.maxprice*0.1
#                elif buyerask <= self.askingprice:
#                    if buyerask > 0:
#                        reward += 0.5#*(self.askingprice - buyerask)#(self.askingprice - buyerask)/self.askingprice*0.1
#                elif (buyerask <= self.maxprice and buyerask > self.askingprice):
#                    reward += 0.5
                    
                
            if buyerask <= 0:
                reward += -2
                
        else:
#            if buyerask == sellerask:
#                reward += 1
#            
##                reward += det_factor
#            elif buyerask > sellerask:
#                reward += -1 * abs(buyerask - sellerask)
#            elif buyerask < sellerask:
##                reward += 1/abs(buyerask-sellerask)
#                reward += -1*abs(buyerask-sellerask)
#                
            if buyerask <=0:
                reward += -1

                
        return reward
#        return reward

    def reset(self):
        self.state = (self.askingprice, self.buyeraskingprice, self.timeLeft, self.determination)
        return np.array(self.state)

#    def close(self):
        