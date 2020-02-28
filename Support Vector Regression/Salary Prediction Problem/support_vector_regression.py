# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:39:43 2020

@author: Mehul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''"""Preprocessing the dataset"""'''
#importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,[-1]].values

#feature scaling has to be applied
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
#we need to create to objects of standard scaler class
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)
#since SVM library does not contain scaling , we have to scale the observations before predicting the output.

'''
#splitting the dataset into a training set and a test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
'''

'''"""Building the model"""'''
# support vector regression model 
#create the regressor by importing SVR class from sklearn.svm 
from sklearn.svm import SVR
regressor =SVR(kernel='rbf')
regressor.fit(X,y)

#predicting result using support vector regression
'''
here,
first : the input has to be scaled according to sc_X
second: the output has to be inverse transformed to get the real result
'''
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualising  svr model results
'''Make the curve smoother'''
# the largest point is considered as the outlier point so the svm leaves it  and fits the model to other points
X_grid=np.arange(min(X),max(X),0.1) #making more points to plot 
X_grid=X_grid.reshape((len(X_grid),1)) # reshaping grid formed into matrix
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title(" Support Vector Regresssion Result")
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()