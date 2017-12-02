# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:57:52 2017

@author: momos_000
"""
# Shared utilities for testing various RL schemes on the Sutton Room World
import datetime
import cPickle as pkl
import os.path
import numpy as np
import matplotlib.pyplot as plt
from room_world import *



with open('20171201-1534_training-history.pkl', 'r') as input:
    hist1 = pkl.load(input)

with open('20171201-1536_training-history.pkl', 'r') as input:
    hist2 = pkl.load(input)

# with open('20171201-1544_training-history.pkl', 'r') as input:
#     hist1 = pkl.load(input)

# with open('20171201-1546_training-history.pkl', 'r') as input:
#     hist2 = pkl.load(input)

# with open('melody_index.pckl', 'r') as input:
#     hist3 = pkl.load(input)    
# def test_compare_plot() :
    # hist = np.zeros((50,7))

   

x = [i for i in range(50)]

# plt.plot(x, hist1[:,5], x, hist2[:,5])
plt.plot(x, hist1[:,5])
plt.plot(x, hist2[:,5])
plt.ylabel('Training Steps')
plt.show()
