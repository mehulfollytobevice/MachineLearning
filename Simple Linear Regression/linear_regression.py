# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:43:03 2020

@author: Mehul
"""

# Simple linear regression

# importing some important  libraries 
import numpy as np
import pandas as pd


#importing the dataset
dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#splitting the dataset into a training set and a test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#fitting the regression model to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results
y_pred=regressor.predict(X_test)

#visualising the results of the training set
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Linear Regression Model For Training Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualising the results of the test set
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test,color="red")
# we observe that the test line and the train line are coincident ,
# that is simply because we have used the same linear regression model
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.plot(X_test,y_pred,color="green")
plt.title("Linear Regression Model For Test Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()