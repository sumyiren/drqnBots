#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:18:38 2018

@author: sumyiren
"""

#Seller Environment


import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
from sellerEnv import sellerEnv
from buyerEnv import buyerEnv

import random

class world():

    def __init__(self, nSellers, maxSteps):
        self.askingPrice = 5
        self.minPrice = 3
        self.maxPrice = 6
        self.buyeraskingPrice = 2
        self.nSellers = nSellers
        self.totalTime = maxSteps
        self.nSellers = nSellers
        self.maxBuyerReward = -100
        self.action_space = spaces.Discrete(3) #less, more, the same
        self.buyerEnvs = []
        self.sellerEnvs = []

        for i in range(self.nSellers):
            self.sellerEnvs.append(sellerEnv(self.totalTime, self.askingPrice, self.minPrice, self.buyeraskingPrice))
            determination = 1 #random.randint(1,5)
            self.buyerEnvs.append(buyerEnv(self.totalTime, self.buyeraskingPrice, self.askingPrice, self.maxPrice, determination))

        self.sellerStates = []
        self.buyerStates = []
        self.sellerRewards = []
        self.buyerRewards = []


    def step(self, actions_seller, actions_buyer):
        
        #do seller step first
        for i in range(self.nSellers):
            state, done = self.sellerEnvs[i].step(actions_seller[i], self.buyerStates[i][1])
            self.sellerStates[i] = state

        
        #do buyer step
        for i in range(self.nSellers):
            state, done = self.buyerEnvs[i].step(actions_buyer[i], self.sellerStates[i][0])
            self.buyerStates[i] = state
        
        
        #calc rewards for seller and buyer
        for i in range(self.nSellers):
            
            if not done:
                reward = self.sellerEnvs[i].calcReward(self.buyerStates[i][0], self.buyerStates[i][1], done)
                self.sellerRewards[i] = reward
                reward = self.buyerEnvs[i].calcReward(self.buyerStates[i][0], self.buyerStates[i][1], done)
                self.buyerRewards[i] = reward
            
            #for done case, only dealmakers for highest sellerask value is winner
            else:
                self.calcFinalReward()
                
                
        return self.sellerStates, self.buyerStates, self.sellerRewards, self.buyerRewards, done
        
        
    def calcFinalReward(self):
        maxVal = 0 #max sellerask value
        valPos = None # position of the max sellerask value
        self.sellerRewards = [0]*self.nSellers
        self.buyerRewards = [0]*self.nSellers
        
        for i in range(self.nSellers):
            if self.buyerStates[i][0] == self.buyerStates[i][1]:
                if self.buyerStates[i][0] > maxVal:
                    valPos = i
                    maxVal = self.buyerStates[i][0]

        if valPos is not None:
            reward = self.sellerEnvs[valPos].calcReward(self.buyerStates[valPos][0], self.buyerStates[valPos][1], True)
            self.sellerRewards[valPos] = reward
            reward = self.buyerEnvs[valPos].calcReward(self.buyerStates[valPos][0], self.buyerStates[valPos][1], True)
            self.buyerRewards[valPos] = reward
            
#        print(self.sellerRewards, self.buyerRewards)

        
    def reset(self):
        n1 = 50
        n2 = 100
        n3 = 1
        n4 = 30
        self.askingPrice = random.randint(n1,n2)
        self.buyeraskingPrice = self.askingPrice - random.randint(n3,n1)
        self.minPrice = self.askingPrice - random.randint(n3,n4)
        self.maxPrice = self.buyeraskingPrice + random.randint(n3,n4)
        
        self.sellerEnvs = []
        self.buyerEnvs = []
        for i in range(self.nSellers):
            self.sellerEnvs.append(sellerEnv(self.totalTime, self.askingPrice, self.minPrice, self.buyeraskingPrice))
            determination = random.randint(0,3)
            self.buyerEnvs.append(buyerEnv(self.totalTime, self.buyeraskingPrice, self.askingPrice, self.maxPrice, determination))

        self.sellerStates = []
        self.buyerStates = []
        self.sellerRewards = []
        self.buyerRewards = []
        for i in range(self.nSellers):
            self.sellerStates.append(self.sellerEnvs[i].reset())
            self.buyerStates.append(self.buyerEnvs[i].reset())
            self.sellerRewards.append(0)
            self.buyerRewards.append(0)
        
        return self.sellerStates, self.buyerStates
        

    
        
        
        
        
