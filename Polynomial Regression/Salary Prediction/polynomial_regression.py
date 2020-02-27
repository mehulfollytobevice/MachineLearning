# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:42:15 2020

@author: Mehul
"""

# importing some important  libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

#we dont need to split it since the data is very less and we need to make  very accurate predictions
#splitting the dataset into a training set and a test set
'''
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
'''
#we will build both models and then compare them
#linear regrssion model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#we can twwek the model by changing the degree , make it more accurate
#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
'''it is same as if we are doing linear regression , but with extra independent variables'''
'''polynomial regression is a special case of multiple linear regression'''
lin_reg2.fit(X_poly,y)

#visualising linear regression model results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Linear Regresssion Result")
plt.xlabel("position label")
plt.ylabel("salary")
plt.scatter(X,lin_reg.predict(X),color="blue")
plt.show()

#visualising polynomial regression model results
'''Make the curve smoother'''
X_grid=np.arange(min(X),max(X),0.1) #making more points to plot 
X_grid=X_grid.reshape((len(X_grid),1)) # reshaping grid formed into matrix

plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Polynomial Regresssion Result")
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()

#Suppose X=8.5
#predicting result using linear regression
result_linear=lin_reg.predict(8.5)

#predicting result using polynomial
'''to get result for polynomial regression first we need to fit and transform the value of the independent variable '''
x=poly_reg.fit_transform(8.5)
result_poly=lin_reg2.predict(x)

'''result_linear and result_poly contain the results for the respective methods '''
""" we observe that using polynomial regression gives a much more accurate result"""
