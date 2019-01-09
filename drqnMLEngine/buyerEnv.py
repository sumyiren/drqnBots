#!/usr/bin/env python5
# -*- coding: utf-8 -*-

#Seller Environment


import gym
import numpy as np

class buyerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self, totalTime, askingprice, maxprice):
        self.askingprice = askingprice
        self.maxprice = maxprice
        self.timeLeft = totalTime

        self.viewer = None
        self.state = None, None, self.maxprice, self.timeLeft, False
        

        self.steps_beyond_done = None


        
    def step(self, action, sellerask_new):
        state = self.state
        sellerask, buyerask, maxprice, timeLeft = state
        if action != None:
            plusminus = self.AVAIL_TORQUE[action]
        else:
            plusminus = 0
            
        buyerask += plusminus
        sellerask = sellerask_new
        timeLeft -= 1
        done = timeLeft <= 0 
        done = bool(done)
        self.state = (sellerask, buyerask, maxprice, timeLeft)       
        return np.array(self.state), done

    def calcReward(self, sellerask, buyerask , done):
        
        reward = 0
        
        if done:
            if buyerask == sellerask:
                if buyerask > self.maxprice:
                    reward += 0#-1 * abs(buyerask - self.maxprice)
                elif (buyerask <= self.maxprice and buyerask > self.askingprice):
                    reward += abs(buyerask - self.maxprice)
                elif buyerask <= self.askingprice:
                    reward += 2* abs(buyerask - self.askingprice)
                    
            else:
                reward += -1 * abs(buyerask - sellerask)

                
            if buyerask <= 0:
                reward += -2
                
        else:
#                
            if buyerask <=0:
                reward += -1

                
        return reward

    def reset(self):
        self.state = [self.askingprice, self.askingprice, self.maxprice, self.timeLeft]
        return np.array(self.state)

#    def close(self):
        