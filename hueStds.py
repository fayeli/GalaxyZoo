# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:47:49 2014

@author: manc
"""

import pandas as pd
import skimage.io as io
import numpy as np
import mahotas
from scipy import ndimage

cen = 212 #cenX and cenY for image
 #length of side of square in which to find find standard dev. of hue
length = 60

#calculate hue from RGB
def hue(r,g,b):
    max = np.max([r,g,b])
    min = np.min([r,g,b])
    delta = max - min + 0.00
    hue = 0
    if (delta != 0):
        if (r == max):
            hue = (g-b)/delta
        elif (g == max):
            hue = 2 + (b-r)/delta
        else:
            hue = 4 + (r-g)/delta
        hue *= 60
        if (hue < 0):
            hue += 360
    else:
        hue = -10;
    return hue;
    
#find central galaxy cluster. 2D boolean representation: if pixel is in or not in galaxy
def findGalaxy(image):
    pic = ndimage.gaussian_filter(image, 8)
    T = mahotas.thresholding.otsu(pic)
    labeled,nr_objects = ndimage.label(pic > T)
    return (labeled == labeled[cen][cen])

#main starts here
#initializing: loading images, creating arrays    
images = io.ImageCollection('1*****.jpg')
n = len(images)
indices = np.empty(n)
hueStds = np.empty(n)

#for each image:
for i in range(n):
    currIm = images[i]
    
    inGalaxy = findGalaxy(currIm)

    #true false array representing whether it is in the galaxy
    indices[i] = images.files[i][0:6]
    hues = []
    #add the hue 
    offset = length / 2
    for j in range(cen-offset,212+offset,1):
        for k in range(212-offset,212+offset,1):
            if (inGalaxy[j][k][0]):
                point = currIm[j][k]
                hues.append(hue(point[0],point[1],point[2]))
    hueStds[i] = np.std(hues)

#storing it in csv format
dictionary = {'HueStd':hueStds}
centers = pd.DataFrame(dictionary,index=indices)
centers.to_csv('hueStds.csv')
