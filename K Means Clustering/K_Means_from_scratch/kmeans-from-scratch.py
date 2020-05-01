# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:02:00 2020

@author: Mehul
"""
# importing the libraries
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

#intial trial data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

#making a scatter plot of the trial data
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

#defining k_means class which contains all the methods needed for clustering 
class k_means:
	#intialising the parameters
	def __init__(self,k=2,tol=0.001,max_iter=300):
		self.k=k
		self.tol=tol
		self.max_iter=max_iter
		
	#training method	
	def fit(self,data):
		self.centroids={}
		 
		#assigning random initial centroids
		for i in range(self.k):
			self.centroids[i]=data[i]
		
		#iterations to find the centroids
		for i in range(self.max_iter):
			self.classifications={}
			
			for i in range(self.k):
				self.classifications[i]=[]
			
			# each datapoint is classified according to it's distancs from the centroid
			for featureset in data:
				#calculating the distances from the centroid
				distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				classification=distances.index(min(distances))
				self.classifications[classification].append(featureset)
				
			prev_centroids=dict(self.centroids)
			
			for classification in self.classifications:
				#findind new centroids
				self.centroids[classification]=np.average(self.classifications[classification],axis=0)
			
			optimized=True
			
			#checking whether the calculated centroids have changed or not 
			#if the value of the centroid does not change much , then it means that the outcome has been optimized
			for c in self.centroids:
				original_centroid=prev_centroids[c]
				current_centroid=self.centroids[c]
				
				if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
					optimized=False
				
			if optimized:
				break
				
	# making predictions
	def predict(self,data):
		distances=[np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
		classification=distances.index(min(distances))
		return classification
	

#Testing out the clustering algorithm
clf=k_means()
clf.fit(X)

colors = 10*["g","r","c","b","k"]

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0],
			 clf.centroids[centroid][1] ,marker="o",color="k",s=100,linewidth=4)

for classification in clf.classifications:
	color=colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=100,linewidth=4)
		
plt.show()

#adding unknown points to see how they are classified 	
new_points=np.array([[1,5],[3,4],[8,9],[5,5],[7,2],[3,9]])

for new in new_points:
	classification=clf.predict(new)
	plt.scatter(new[0],new[1],marker='*',color=colors[classification],s=100,linewidth=4)

plt.show()
		