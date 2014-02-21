# elimination.py

import numpy as np
import scipy
import pylab
import pymorph
import mahotas
import skimage.io as io
import pandas as pd
from scipy import ndimage

#Prints roundness of one image on terminal
# def main():
# 	centregalaxy = eliminate(153666) #Galaxyid goes here
# 	print centregalaxy
# 	pylab.imshow(centregalaxy)
# 	pylab.show()

#Creates csv file for roundness of all images
def main():
	images = io.ImageCollection('1*****.jpg')
	n = len(images)
	indices = np.empty(n)
	roundness = np.empty(n)
	for i in range(n):
		indices[i] = images.files[i][0:6]
		roundness[i] = eliminate(indices[i])
	R = {'Roundness':roundness}
	table = pd.DataFrame(R,index=indices)
	table.to_csv('roundness.csv')
# Returns a 3 dimentional array True/False or Returns roundness
# Its shape is (424,424,3)

def eliminate(galaxyid):
	filename = str(galaxyid)[0:6]+'.jpg'
	#First, we load the image
	pic = mahotas.imread(filename)

	#Smooth the image using a Gaussian filter
	picf = ndimage.gaussian_filter(pic, 8)
	T = mahotas.thresholding.otsu(picf)

	#Label each blob in the image
	labeled,nr_objects = ndimage.label(picf > T)

	#Getting rid of blobs that are not our target galaxy
	centre = (labeled==labeled[212,212,1])
	perimiter = mahotas.labeled.bwperim(centre)
	p = np.sum(perimiter)
	area = np.sum(centre)
	roundness = (p*p/(4*np.pi))/area
	return roundness #The lower 'roundness' is, the rounder the galaxy is
	#return centre

if __name__ == "__main__":
    main()