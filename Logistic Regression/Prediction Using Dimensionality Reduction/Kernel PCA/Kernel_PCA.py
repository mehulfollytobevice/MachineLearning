# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:49:12 2020

@author: Mehul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Social_Network_Ads.csv")
'''
for X we are using only two factors , 
age and salary to apply logistic regression and find out who purchases the product and who does not 
'''
X=dataset.iloc[:,2:4].values
y=dataset.iloc[:,-1].values

#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#y consists of categorical values so we do not need to scale it

'''
Since the data is not properly linearly seperable,
we apply Kernel PCA to first map the dataset to a higher dimension and then
apply PCA on this new dataset to reduce it's dimension by extracting
new principle components.
'''
#Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca=KernelPCA(n_components=2,kernel='rbf')
X_train=kpca.fit_transform(X_train)
X_test=kpca.transform(X_test)

#Building the classifier
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
#it learns the correlation between X_train and y_train
classifier.fit(X_train,y_train)

#testing on the test set
y_pred=classifier.predict(X_test)

#making the confusion matrix
'''the diagonal elements show how many of the elements matched in the arrays of y_test and y_pred'''
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
pred_true=cm[0,0]+cm[1,1]
pred_false=cm[0,1]+cm[1,0]
print("correct predictions:"+str(pred_true))
print("incorrect predictions:"+str(pred_false))

#visualising the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
'''meshgrid forms a grid or a matrix of all the pixels displayed on the screen  '''
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
'''
this function makes a countour between the red and the green , 
it paints the pixels in the region in the graph as red and green according to the prediction 
'''
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#visualising the test set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

'''
Notes:
	1.Kernel PCA uses the kernel trick to map the dataset to a higher dimension.
	2.This is an unsupervised algorithm.
	3.This works good for seperating non-linear datasets using linear classifiers.
'''