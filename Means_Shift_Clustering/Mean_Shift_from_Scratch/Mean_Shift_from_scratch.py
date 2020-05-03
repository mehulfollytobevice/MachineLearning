# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:47:41 2020

@author: Mehul
"""
#importing libraries
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import  make_blobs

#generating random samples for clustering 
X,y=make_blobs(n_samples=50,centers=3,n_features=2)

colors = 10*["g","r","c","b","k"]

#class Mean_Shift ,containing all the methods for training and predicting 
class Mean_Shift:
	def __init__(self,bandwidth=None,bandwidth_step=100):
		self.bandwidth=bandwidth
		self.bandwidth_step=bandwidth_step
		
	def fit(self,data):
		centroids={}
		 
		#initial centroid position 
		if self.bandwidth==None:
			all_data_centroid=np.average(data,axis=0)
			all_data_norm=np.linalg.norm(all_data_centroid)#distance from the origin
			self.bandwidth=all_data_norm/self.bandwidth_step
		
		
		for i in range(len(data)):
			centroids[i]=data[i] #initially all the points are centroids themselves
		
		weights=[i for i in range(self.bandwidth_step)][::-1] #assigning weights
		
		while True: #infinite loop , runs until convergence(optimisation) occurs
			new_centroids=[]
			for i in centroids:
				in_bandwidth=[]
				centroid=centroids[i] # iterating through all the centroids
				
				for featureset in data:
					distance=np.linalg.norm(featureset-centroid)#finding all the distances from a particular centroid
					if distance==0:
						distance=0.00000000001
						
					weight_index=int (distance/self.bandwidth) #tells us how close the point is
					
					if weight_index>self.bandwidth_step-1:
						weight_index=self.bandwidth_step-1 # after a certain point weight does not decrease
					
					to_add=(weights[weight_index]**2)*[featureset]
					in_bandwidth+=to_add #multiplying weight prioritizes closer points than the further ones
				
				new_centroid=np.average(in_bandwidth,axis=0) #new centroid in the average of the points within the bandwidth
				new_centroids.append(tuple(new_centroid))
				
			uniques=sorted(list(set(new_centroids)))
			
			to_pop=[]
			
			#here we start removing centroids that come within each other's bandwidth
			for i in uniques:
				for ii in uniques:
					if i==ii:
						pass
					elif np.linalg.norm(np.array(i)-np.array(ii))<=self.bandwidth:
						to_pop.append(ii)
						break
						
			for i in to_pop:
				try:
					uniques.remove(i)
				except:
					pass
			
			#starting to check for convergence
			prev_centroids=dict(centroids)
			#if the previous centroids are equal to the current centroids then the algorithm is optimized otherwise the while loop keeps on going
			centroids={}
			for i in range(len(uniques)):
				centroids[i]=np.array(uniques[i])
				
			optimized=True
			
			for i in centroids:
				if not np.array_equal(centroids[i],prev_centroids[i]): #checking for convergence 
					optimized=False
				if not optimized:
					break
			if optimized:
				break
		
		#classifying points on the basis of the final centroids 
		self.centroids=centroids
		self.classifications={}
		
		for i in range(len(self.centroids)):
			self.classifications[i]=[]
			
		for featureset in data:
			distances =[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
			classification=(distances.index(min(distances)))
			self.classifications[classification].append(featureset)
	
#predict method	
	def predict(self,data):
		distances =[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification=distances.index(min(distances))
		return classification

			
#Training and testing the algorithm
clf=Mean_Shift()
clf.fit(X)

centroids=clf.centroids
print(centroids)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker = "o", color=color, s=150, linewidths = 2)

for i in centroids:
	plt.scatter(centroids[i][0],centroids[i][1],color='k',marker='*',linewidth=5)

plt.show()
			
			
			