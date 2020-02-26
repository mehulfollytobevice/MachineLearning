# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:01:59 2020

@author: Mehul
"""

# multiple linear regression
# importing some important  libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Catagorical  data or encoding 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
# so we get the encoded values but there might be problems, that is why we do dummy coding

#Dummy coding using OneHotEncoder
#X
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap
#Although sklearn takes care of  dummy variable trap , some softwares don't
'''X=X[:,1:]'''


#splitting the dataset into a training set and a test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#feature scaling is not needed since the library takes care of it 
#fitting linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the results for the test set
y_pred=regressor.predict(X_test)
#it is noted that the predicted values are very close to the actual test values 
#thus there is a linear relation between the different variables and the dependent variable

#since the above model is not the most efficient model , we will use backward elimination 
#backward variables will help us keep only those variables that are statistically significant
#backward elimination preparation
import statsmodels.formula.api as sm
#the above library does not take into account the constant b0 in the regression equation
#so we add an array of ones to compensate it 
#first we remove one extra dummy variable column
X=X[:,2:]
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#initialising the optimal set/matrix of features
X_opt=X[:,[0,1,2,3,4,5]]

#Backward Elimination
'''
Step 1:Select a significance level to stay in the model (eg. SL = 0.05)
Step 2:Fit the model with all possible predictors
Step 3:Consider the predictor with the highest P-value. If P>SL, go to Step 4.
Step 4:Remove the predictor
Step 5:Fit the model without this variable and repeat the steps 3,4,5 until the condition P>SL becomes false.
'''

#backward elimination
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary()

#removing extra added columnn of 1's 
X_opt=X[:,[3]]

#applying regression
from sklearn.cross_validation import train_test_split
X_train2,X_test2,y_train2,y_test2=train_test_split(X_opt,y,test_size=0.2,random_state=0)


#feature scaling is not needed since the library takes care of it 
#fitting linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train2,y_train2)

#predicting the results for the test set
y_pred2=regressor.predict(X_test2)

print("Invest in the startup with the maximum predicted profit.")


