#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:57:50 2019

@author: sumyiren
"""
from testdrqnUser import testDrqn


test = testDrqn()
test.restart()
a = test.stepAction(0)
print(a)
