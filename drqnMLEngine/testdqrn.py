#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:52:18 2018

@author: sumyiren
"""
from testClass import testClass


test = testClass()
test.restart()
done = False

while not done:
    obs_seller, obs_buyer, done = test.stepAction()
    print('-------------------------------------')
    for i in range(test.nSellers):
        print('nSeller:'+str(i))
        print('SellerAsk = ' +str(obs_seller[i])+ 'BuyerAsk = ' + str(obs_buyer[i]))

    
    