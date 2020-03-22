# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:46:33 2020

@author: Mehul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset 
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimum numbers of clusters 
from sklearn.cluster import KMeans
wcss=[]
'''using iteration to find optimum number od clusters'''
for i in range(1,11):
	kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("the number of clusters")
plt.ylabel("wcss")
plt.show()

'''
here we find that the optimal number of clusters is 5
thus we find that there are 5 segments of the customers
'''

#applying k-means on the optimal number of clusters
kmeans=KMeans(n_clusters=5,init="k-means++",n_init=10,max_iter=300,random_state=0)
'''this returns a vector of values which shows which cluster each point of the dataset belongs to'''
y_kmeans=kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="blue",label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="green",label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="cyan",label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="magenta",label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Centroids")
plt.title("Clusters")
plt.xlabel("Annual Income(k$)")
plt.ylabel("spending score")
plt.legend()
plt.show()

#re-visualising the clusters after analysing the plot
''' after analysing the plot we find out that the kinds of segments in the data '''
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label='Careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="blue",label='Standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="green",label='Target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="cyan",label='Careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="magenta",label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Centroids")
plt.title("Clusters of clients")
plt.xlabel("Annual Income(k$)")
plt.ylabel("spending score")
plt.legend()
plt.show()
