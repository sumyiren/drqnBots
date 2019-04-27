#!/usr/bin/env python5
# -*- coding: utf-8 -*-

#Seller Environment


import gym
import numpy as np
import math
class buyerEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    AVAIL_TORQUE = [-1., 0., +1]

    def __init__(self, totalTime, sellerStartingPrice, buyerStartingPrice, maxprice):
        self.sellerStartingPrice = sellerStartingPrice
        self.buyerStartingPrice = buyerStartingPrice
        self.maxprice = maxprice
        self.timeLeft = totalTime

        self.viewer = None
        self.state = None, None, self.maxprice, self.timeLeft, False
        

        self.steps_beyond_done = None


        
    def step(self, action_buyer, action_seller):
        state = self.state
        sellerask, buyerask, maxprice, timeLeft = state
        if action_buyer != None:
            plusminus_buyer = self.AVAIL_TORQUE[action_buyer]
            plusminus_seller = self.AVAIL_TORQUE[action_seller]
        else:
            plusminus_buyer = 0
            plusminus_seller = 0
            
        buyerask += plusminus_buyer
        sellerask += plusminus_seller
        timeLeft -= 1
        done = timeLeft <= 0 
        done = bool(done)
        self.state = (sellerask, buyerask, maxprice, timeLeft)       
        return np.array(self.state), done

    def calcReward(self, sellerask, buyerask , done):
        
        reward = 0
        
        if done:
            if abs(buyerask - sellerask) <= 1: #avoid negative rewards here
#                if buyerask > self.maxprice:
#                    reward += 0#-1 * abs(buyerask - self.maxprice)
#                elif (buyerask <= self.maxprice and buyerask > self.buyerbuyerStartingPrice):
#                    reward += abs(buyerask - self.maxprice)
#                elif buyerask <= self.buyerStartingPrice:
#                    reward += 2* abs(buyerask - self.buyerStartingPrice)
                
                if buyerask <= self.maxprice:
                    reward += 3 + 2*abs(buyerask - self.maxprice)
                else:
                    reward += 3*math.exp((self.maxprice - buyerask)/10)

                   
            else:
                reward += -1 * abs(buyerask - sellerask)

                
            if buyerask <= 0:
                reward += -2
                
        else:
#                
            if buyerask > self.maxprice:
                reward += -1
            if buyerask <=0:
                reward += -1

                
        return reward

    def reset(self):
        self.state = [self.sellerStartingPrice, self.buyerStartingPrice, self.maxprice, self.timeLeft]
        return np.array(self.state)

#    def close(self):
        