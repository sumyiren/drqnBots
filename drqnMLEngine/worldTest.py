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
from sellerEnv import sellerEnv
from buyerEnv import buyerEnv

import random

class world():

    def __init__(self, nSellers, maxSteps, teamSpirit):
        self.askingPrice = 5.0
        self.nSellers = nSellers
        self.totalTime = maxSteps
        self.nSellers = nSellers
        self.maxBuyerReward = -100
        self.action_space = spaces.Discrete(3) #less, more, the same
        self.buyerEnvs = []
        self.sellerEnvs = []
        self.teamSpirit = teamSpirit

        for i in range(self.nSellers):
            self.sellerEnvs.append(sellerEnv(self.totalTime, self.askingPrice, self.askingPrice, 0))
            self.buyerEnvs.append(buyerEnv(self.totalTime, self.askingPrice, self.askingPrice, 0))

    def flattenList(self, list):
        flat_list = [item for sublist in list for item in sublist]
        return flat_list
        
#    def getSellerStackStates(self):
#        sellerStackStates = []
#        for i in range(self.nSellers):
#            temp = []
#            temp.extend(self.sellerStates[i][:])
#            for j in range(self.nSellers):
#                if i != j:
#                    temp.extend(self.sellerStates[j][:])
#            sellerStackStates.append(temp)
#        return sellerStackStates
            
        
    # now in the form [seller, buyer, seller, buyer, minPrice, timeRemaining]
    def getSellerStackStates(self):
        sellerStackStates = []
        for i in range(self.nSellers):
            temp = []
            temp.extend(self.sellerStates[i][0:2])
            for j in range(self.nSellers):
                if i != j:
                    temp.extend(self.sellerStates[j][0:2])
            temp.extend(self.sellerStates[i][-2:])
            sellerStackStates.append(temp)
        return sellerStackStates
        

    def step(self, actions_seller, actions_buyer):
        
        #do seller step first
        for i in range(self.nSellers):
            state, done = self.sellerEnvs[i].step(actions_seller[i], actions_buyer[i])
            self.sellerStates[i] = state

        self.sellerStackStates = self.getSellerStackStates()
        #do buyer step
        for i in range(self.nSellers):
            state, done = self.buyerEnvs[i].step(actions_buyer[i], actions_seller[i])
            self.buyerStates[i] = state

        
        #calc rewards for seller and buyer
        for i in range(self.nSellers):
            
            reward = self.sellerEnvs[i].calcReward(self.buyerStates[i][0], self.buyerStates[i][1], done)
            self.sellerRewards[i] = reward
            reward = self.buyerEnvs[i].calcReward(self.buyerStates[i][0], self.buyerStates[i][1], done)
            self.buyerRewards[i] = reward

        if done: 
            self.sellerRewards = self.calcFinalSellerReward(self.sellerRewards)
                
        return self.sellerStackStates, self.buyerStates, self.sellerRewards, self.buyerRewards, done
        
        
    #deprecated - unused
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

#    def calcFinalSellerReward(self, sellerReward): #actually no need return since pbr, but for clarity
#        maxSellerReward = max(sellerReward)
#        if maxSellerReward >= 0:
#            for i in range(len(sellerReward)):
#                sellerReward[i] = maxSellerReward
#        return sellerReward

    def calcFinalSellerReward(self, sellerReward): #annealed - when near end, the top seller benefits everyone
#        avgSellerReward = np.average(sellerReward)
        maxSellerReward = max(sellerReward)
        for i in range(len(sellerReward)):
            sellerReward[i] = self.teamSpirit*maxSellerReward + (1-self.teamSpirit)*sellerReward[i]
        return sellerReward
        
    def reset(self):
        n1 = 1
        n2 = 150.0
        
        self.sellerStartingPrice = 100
        self.minPrice = 80
        
        self.sellerEnvs = []
        self.buyerEnvs = []
        self.buyerStartingPrice = 40
        for i in range(self.nSellers):
            
            self.maxPrice = self.buyerStartingPrice + random.randint(0,n2-self.buyerStartingPrice+1)
            
            self.sellerEnvs.append(sellerEnv(self.totalTime, self.sellerStartingPrice, self.buyerStartingPrice, self.minPrice))
            self.buyerEnvs.append(buyerEnv(self.totalTime, self.sellerStartingPrice, self.buyerStartingPrice, self.maxPrice))

        self.sellerStates = []
        self.sellerStackStates = []
        self.buyerStates = []
        self.sellerRewards = []
        self.buyerRewards = []
        for i in range(self.nSellers):
            self.sellerStates.append(self.sellerEnvs[i].reset())
            self.buyerStates.append(self.buyerEnvs[i].reset())
            self.sellerRewards.append(0)
            self.buyerRewards.append(0)
        
        self.sellerStackStates = self.getSellerStackStates()
        
        return self.sellerStackStates, self.buyerStates
        

    
        
        
        
        
