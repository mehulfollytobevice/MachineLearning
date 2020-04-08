# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:14:44 2020

@author: Mehul
"""

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#import the dataset 
dataset=pd.read_csv('bottle.csv')

#extracting the essential features
dataset=dataset[['T_degC','Salnty']]

#visualising the data before fitting the line and before deleting outliers 
plt.scatter(dataset['T_degC'],dataset['Salnty'])
plt.show()

#dealing with missing values
sum_temperatures=dataset.sum(axis = 0, skipna = True)['T_degC']
number_of_observations=dataset[dataset['T_degC']>=0].shape[0]
average=sum_temperatures/number_of_observations
dataset["T_degC"].fillna(average, inplace = True)

sum_temperatures=dataset.sum(axis = 0, skipna = True)['Salnty']
number_of_observations=dataset[dataset['Salnty']>=0].shape[0]
average=sum_temperatures/number_of_observations
dataset["Salnty"].fillna(average, inplace = True)

#visualising the data before fitting the line and before deleting outliers 
plt.scatter(dataset['T_degC'],dataset['Salnty'])
plt.show()

#deleting the outliers so that the model could be fitted better
dataset=dataset.drop(dataset[(dataset['Salnty']<31.5)].index)
dataset=dataset.drop(dataset[(dataset['Salnty']>36)].index)
dataset=dataset.drop(dataset[(dataset['Salnty']>35) & (dataset['T_degC']<11.5)].index)
dataset=dataset.drop(dataset[(dataset['Salnty']<33.72) & (dataset['T_degC']<4.5)].index)
dataset=dataset.drop(dataset[(dataset['Salnty']>35) & (dataset['T_degC']<11.2)].index)

#visualising the data before fitting the line and after deleting outliers 
plt.scatter(dataset['T_degC'],dataset['Salnty'])
plt.show()

#extracting the matrix of features and the dependent variable
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,-1].values

#splitting the dataset into training set and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#building the regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the outcome 
y_pred=regressor.predict(X_test)

#visualise the results
plt.scatter(dataset['T_degC'],dataset['Salnty'],color='red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Linear Regression Model For Training Set")
plt.xlabel("Temperature")
plt.ylabel("Salanity of water")
plt.show()

#lets see the summary about the regression line
import statsmodels.regression.linear_model as sm
regressor_ols=sm.OLS(endog=y,exog=X).fit()
regressor_ols.summary()
#adjusted R-squared is around .863 which is pretty good 
#it means that our best fit line explains 86% of the variation

#using a polynomial model as an experiment
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=7)
X_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
'''it is same as if we are doing linear regression , but with extra independent variables'''
'''polynomial regression is a special case of multiple linear regression'''
lin_reg2.fit(X_poly,y)

#making the predictions
y_pred2=lin_reg2.predict(X_poly)

#visualising polynomial regression model results
'''Make the curve smoother'''
X_grid=np.arange(min(X),max(X),0.1) #making more points to plot 
X_grid=X_grid.reshape((len(X_grid),1)) # reshaping grid formed into matrix

plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Polynomial Regresssion Result")
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.show()

#lets see the summary for the polynomial curve
import statsmodels.regression.linear_model as sm
regressor_ols=sm.OLS(endog=y,exog=X_poly).fit()
regressor_ols.summary()
