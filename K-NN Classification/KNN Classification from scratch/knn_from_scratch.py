# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:03:57 2020

@author: Mehul
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
from matplotlib import style
from collections import Counter
from math import sqrt
style.use('fivethirtyeight')

#defining knn function
def k_nearest_neighbors(data,predict,k=3):
	distances=[]
	if(len(data)>=k):
		#this is not an error it is just a warning , the algorithm still works 
		warnings.warn('The value of k is less than the number of voting groups.')
    
	for group in data:
		#data is a dictionary of lists with different groups of classes 
		for features in data[group]:
			#features represent the points in the dataset
			
			#original way
			#euclidean_distance=sqrt((features[0]-predict[0])**2+(features[1]-predict[1])**2)
			
			#faster way
			euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance,group])
	
	#once we have the distances we dont care about them
	#we populate the list of votes which has the top k neighbors to the prediction point 
	votes=[i[1] for i in sorted(distances)[:k] ]
	#using counter we calculate the most common out of the nearest neighbors
	vote_result=Counter(votes).most_common(1)[0][0]
	
	#we can also give our confidence,confidence is the probability of your prediction being right
	#confidence=Counter(votes).most_common(1)[0][1]/k
	
	return vote_result

def accuracy_of_result(train_set,test_set):
	#intialising 
	correct=0
	total=0
	
	#testing and finding accuracy
	for group in test_set:
		for data in test_set[group]:
			#iterating through all the data in a class 
			result=k_nearest_neighbors(train_set,data,k=5)
			if (group==result):
				correct=correct+1
			total=total+1
	accuracy=correct/total
	return accuracy

''''
#trial data
#our data is in form of dictionary of lists
dataset={'k':[[1,2],[2,3,],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features=[5,7]

#plotting the data
plt.scatter(new_features[0],new_features[1],s=50)
for i in dataset:
	for j in dataset[i]:
		print(j)
		plt.scatter(j[0],j[1],s=100,color=i)

#applying knn model
result=k_nearest_neighbors(dataset,new_features,k=3)#result represents the class the prediction point belongs to 

#plotting the prediction
plt.scatter(new_features[0],new_features[1],s=50,color=result)
for i in dataset:
	for j in dataset[i]:
		print(j)
		plt.scatter(j[0],j[1],s=100,color=i)
'''

#Implmenting the model on the test dataset

#importing the dataset
dataset=pd.read_csv('breast-cancer-wisconsin.data.txt')

#replacing missing instances with large numbers 
dataset.replace('?',-99999,inplace=True)
dataset.drop(['id'],1,inplace=True)
dataset=dataset.astype(float).values.tolist()

#shuffling to data to include some randomness
#this does not change the raltionship between the data
#this is what can be used for cross-validation 
random.shuffle(dataset)

#splitting the dataset into test set and train set
test_size=0.2

#the train set and the test set are dictionary of lists
train_set={2:[],4:[]}
test_set={2:[],4:[]}

#slicing the data into train_data and test_data
train_data=dataset[:-int(test_size*len(dataset))] #all the data upto the last 20%
test_data=dataset[-int(test_size*len(dataset)):] #the last 20%

#populating the dictionary
#here we take the data from the train_data and the test_data and use it to populate our dictionaries

for i in train_data:
	train_set[i[-1]].append(i[:-1])# i[-1] represents the class of the particular row

for i in test_data:
	test_set[i[-1]].append(i[:-1])# i[-1] represents the class of the particular row

#getting the accuracy of our knn model on the dataset
print('Accuracy of the result:',accuracy_of_result(train_set,test_set))