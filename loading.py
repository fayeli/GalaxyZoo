# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:35:02 2014

@author: manc
"""

import pandas as pd
import skimage.io as io
import numpy as np

images = io.ImageCollection('1*****.jpg')
n = len(images)
indices = np.empty(n)
R = np.empty(n)
G = np.empty(n)
B = np.empty(n)
for i in range(n):
    center = images[i][212][212]
    indices[i] = images.files[i][0:6]
    R[i] = center[0]
    G[i] = center[1]
    B[i] = center[2]
    
colors = {'R':R, 'G':G, 'B':B}
centers = pd.DataFrame(colors,index=indices)
centers.to_csv('colors.csv')