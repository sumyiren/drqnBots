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
from drqnMLEngine.sellerEnv import sellerEnv
from drqnMLEngine.buyerEnv import buyerEnv

import random

class world():

    def __init__(self, nSellers, maxSteps, teamSpirit):
        self.askingPrice = 5.0
        self.nSellers = nSellers
        self.totalTime = maxSteps
        self.maxRange = maxSteps
        self.nSellers = nSellers
        self.maxBuyerReward = -100
        self.action_space = spaces.Discrete(3) #less, more, the same
        self.buyerEnvs = []
        self.sellerEnvs = []
        self.teamSpirit = teamSpirit

        for i in range(self.nSellers):
            self.sellerEnvs.append(sellerEnv(self.totalTime, self.askingPrice, self.askingPrice, 0))
            self.buyerEnvs.append(buyerEnv(self.totalTime, self.askingPrice, self.askingPrice, 0))

#    def flattenList(self, list):
#        flat_list = [item for sublist in list for item in sublist]
#        return flat_list
        
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
    
    def getBuyerStackStates(self):
        buyerStackStates = []
        for i in range(self.nSellers):
            temp = []
            temp.extend(self.buyerStates[i][0:2])
            for j in range(self.nSellers):
                if i != j:
                    temp.extend(self.buyerStates[j][1:2])
            temp.extend(self.buyerStates[i][-2:])
            buyerStackStates.append(temp)
        return buyerStackStates
        

    def step(self, actions_seller, actions_buyer):
        
        #do seller step first
        for i in range(self.nSellers):
            state, done = self.sellerEnvs[i].step(actions_seller[i], actions_buyer[i])
            self.sellerStates[i] = state

        #do buyer step
        for i in range(self.nSellers):
            state, done = self.buyerEnvs[i].step(actions_buyer[i], actions_seller[i])
            self.buyerStates[i] = state

        self.sellerStackStates = self.getSellerStackStates()
        self.buyerStackStates = self.getBuyerStackStates()   
        #calc rewards for seller and buyer
        for i in range(self.nSellers):
            self.sellerRewards[i] = self.calcSellerReward(self.sellerEnvs[i], self.buyerStates[i][0], self.buyerStates[i][1], done)
            self.buyerRewards[i] = self.calcBuyerReward(self.buyerEnvs[i], self.buyerStates[i][0], self.buyerStates[i][1], done)

        if done: 
            self.sellerRewards = self.calcFinalSellerReward(self.sellerRewards)
            self.buyerRewards = self.calcFinalBuyerReward(self.buyerRewards, self.buyerStates )
                
        return self.sellerStackStates, self.buyerStackStates, self.sellerRewards, self.buyerRewards, done
        

    def calcSellerReward(self, sellerEnv,  sellerask, buyerask, done):
        minPrice = sellerEnv.minPrice
        reward = 0
        
        if done:
            if abs(sellerask - buyerask) <= 2 : #maybe 2? - deal made
                
                if sellerask >= minPrice:
                    reward += 2*(sellerask - minPrice)
                else:
                    reward += - 0.5* abs(sellerask - minPrice)
                    
            else:
                reward += -1*abs(sellerask-buyerask)
                reward += 0.5* abs(sellerask - minPrice)
            
            if sellerask <=0:
                reward = -1000
                
        else: #do reward shaping here
#            if sellerask < minPrice:
#                reward += -1
#            if sellerask <=0:
#                reward += -10
            shaping = -0.5*abs(sellerask-buyerask) # And ten points for legs contact, the idea is if you
            shaping += 0.25*(sellerask - minPrice)
        
            if (sellerask - buyerask) < 0:
                shaping += -10
                
            if sellerask <=0:
                shaping += -10
                
            if sellerEnv.prev_shaping is not None:
                reward = shaping - sellerEnv.prev_shaping
            sellerEnv.prev_shaping = shaping
            

        return reward
        

    def calcBuyerReward(self, buyerEnv, sellerask, buyerask, done):
        maxPrice = buyerEnv.maxPrice
        reward = 0
        
        
        if done:
            if abs(sellerask - buyerask) <= 2 : 
                
                if buyerask <= maxPrice:
                    reward += 2*(maxPrice - buyerask)
                else:
                    reward += - 0.5* abs(maxPrice - buyerask) 
                    
            else:
                reward += -1*abs(sellerask-buyerask)
                reward += 0.5* abs(maxPrice - buyerask)
                
            if buyerask <=0:
                reward = -1000
        else:
            
            shaping = -0.5*abs(sellerask-buyerask) # And ten points for legs contact, the idea is if you
            shaping += 0.25*(maxPrice - buyerask)
            
            if buyerask <=0:
                shaping += -10
                
            if buyerEnv.prev_shaping is not None:
                reward = shaping - buyerEnv.prev_shaping
            buyerEnv.prev_shaping = shaping
        return reward
        
    
    def calcFinalBuyerReward(self, buyerRewards, buyerStates):
        currLowestBuyerAsk = 10000
        currHighestReward = None
        for i in range(len(buyerRewards)):
            if buyerRewards[i] > 0:
                currHighestReward = buyerRewards[i]
                buyerRewards[i] = 0
                if buyerStates[i][1] < currLowestBuyerAsk:
                    currLowestBuyerAsk = buyerStates[i][1]
                    buyerRewards[i] = currHighestReward
                    
        return buyerRewards


    def calcFinalSellerReward(self, sellerRewards):
        maxSellerReward = max(sellerRewards)
        for i in range(len(sellerRewards)):
            sellerRewards[i] = self.teamSpirit*maxSellerReward + (1-self.teamSpirit)*sellerRewards[i]
        return sellerRewards
        
#    def calcFinalSellerReward(self, sellerRewards, sellerEnvs, buyerEnvs): 
#        goodDealRewards = []
#        for i in range(len(sellerRewards)):
#            goodDeal = buyerEnvs[i].maxPrice >= sellerEnvs[i].minPrice
#            if goodDeal:
#                goodDealRewards.append(sellerRewards[i])
#        
#        #annealed - when near end, the top seller benefits everyone - but if no deal, negative reward high
##        avgSellerReward = np.average(sellerReward)
#        if len(goodDealRewards):
#            maxSellerReward = max(goodDealRewards)
#            minSellerReward = min(goodDealRewards)
#            if maxSellerReward > 0:
#                for i in range(len(sellerRewards)):
#                    sellerRewards[i] = self.teamSpirit*maxSellerReward + (1-self.teamSpirit)*sellerRewards[i]
#            else:
#                #punish seller hard if no deal made
#                for i in range(len(sellerRewards)):
#                    sellerRewards[i] = self.teamSpirit*minSellerReward + (1-self.teamSpirit)*sellerRewards[i]
#        return sellerRewards
        
    def reset(self):
        n1 = 2
        n2 = 200
        
        self.minPrice = random.randint(n1,n2-1)
        self.sellerStartingPrice = self.minPrice
        
        self.sellerEnvs = []
        self.buyerEnvs = []
        for i in range(self.nSellers):
            
            self.maxPrice = random.randint(self.minPrice,n2)
            self.buyerStartingPrice = random.randint(0, self.minPrice - 1)
            
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
        self.buyerStackStates = self.getBuyerStackStates()
        
        return self.sellerStackStates, self.buyerStackStates
        

    
        
        
        
        
