# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:15:00 2020

@author: Mehul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Wine.csv")

X=dataset.iloc[:,0:13].values
y=dataset.iloc[:,-1].values

#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#feature scaling
#this must be applied when we apply PCA or LDA
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#y consists of categorical values so we do not need to scale it

#Applying the LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)#to visualise the data , we are taking 2 linear discriminants
X_train=lda.fit_transform(X_train,y_train)
'''we don't need y_test since the structure and relevant info about the dependent variable is already fitted using y_train'''
X_test=lda.transform(X_test)

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

#visualising the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
'''meshgrid forms a grid or a matrix of all the pixels displayed on the screen  '''
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

#visualising the test set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


'''
Notes:
	1. LDA is quite vulnerable to outliers , and concentrates on the points towards the middle.
	2.It is a supervised algorithm so it requires the structure of the dependent variable.
	3.The prediction boundary btween two classes is equidistant to the closest points on each direction.
	4.Since LDA is a linear method it cannot be applied to non-linear ML models.
'''