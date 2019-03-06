#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:25:23 2019

@author: sumyiren
"""
import pickle

with open('multitestdqrn.pickle', 'rb') as handle:
    testRes = pickle.load(handle) 

startArrays = testRes['start']
endArrays = testRes['end']  

nTest = len(endArrays)
nSellers = len(endArrays[0]['buyer'])

start_buyer = []
start_seller = []
end_buyer = []
end_seller = []

for i in range(nTest):
    start_buyer.append(startArrays[i]['buyer'])   
    end_buyer.append(endArrays[i]['buyer'])  
    start_seller.append(startArrays[i]['seller'])  
    end_seller.append(endArrays[i]['seller'])  


#deals made
def getDealMadeListIndex(obs_seller_list):
    dealMadeListIndex = []
    for i in range(nTest):
        for j in range(nSellers):
            if abs(obs_seller_list[i][j*2]-obs_seller_list[i][j*2+1]) <= 1:
                dealMadeListIndex.append(i)
                break
    return dealMadeListIndex

obs_seller_list = []
for i in range(nTest):
    obs_seller_list.append(end_seller[i][0])
    
dealMadeListIndex = getDealMadeListIndex(obs_seller_list)
print('Total Deals Made out of '+str(nTest)+ ' tests: '+str(len(dealMadeListIndex)))

#e.g for 3 nsellers - [1,2,0] means lowest bidder has 1 deal, second lowest has 2 deals, highest bidder made 0 deals
#if 2 deals made in 1 test, the highest deal for seller take precedence
def allocateHighestDealLocation(obs_buyer):
    obs_buyer_sorted = sorted(obs_buyer, key=lambda x: x[2])
    highestSellerVal = None
    highestBidLocation = None
    
    for i in range(nSellers):
        if abs(obs_buyer_sorted[i][0]-obs_buyer_sorted[i][1]) <= 1:
            if highestSellerVal is None or obs_buyer_sorted[i][0] > highestSellerVal:
                highestSellerVal = obs_buyer_sorted[i][0]
                highestBidLocation = i
#    print(obs_buyer_sorted)
#    print(highestBidLocation)
    return highestBidLocation

bidLocationList = [0]*nSellers
for index in dealMadeListIndex:
    bidLocation = allocateHighestDealLocation(end_buyer[index])
    bidLocationList[bidLocation] += 1

#1 being lowest bidder
for i in range(nSellers):
    print('Deals sold to '+str(i+1)+ ' Bidder: '+str(bidLocationList[i]))


#check for bias within trained sellers - i.e position 1 seller trained to closed deals out more often than not even when unfavourable
def allocateTrainedBotLocation(obs_buyer):
    highestSellerVal = None
    trainedBotLocation = None
    for i in range(nSellers):
        if abs(obs_buyer[i][0]-obs_buyer[i][1]) <= 1:
            if highestSellerVal is None or obs_buyer[i][0] > highestSellerVal:
                highestSellerVal = obs_buyer[i][0]
                trainedBotLocation = i
    return trainedBotLocation

trainedBotLocationList = [0]*nSellers
for index in dealMadeListIndex:
    botLocation = allocateTrainedBotLocation(end_buyer[index])
    trainedBotLocationList[botLocation] += 1

for i in range(nSellers):
    print('Deals obtained by Bot '+str(i+1)+ ': '+str(trainedBotLocationList[i]))

#seller favoured - above starting price


#buyer favoured - below starting price

#seller average earn - for favoured


#buyer average earn -  for favoured