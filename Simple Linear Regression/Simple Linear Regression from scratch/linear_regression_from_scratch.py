# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:49:15 2020

@author: Mehul
"""
from statistics import mean
import numpy as np
import random

#xs=np.array([1,2,3,4,5,6,7],dtype=np.float64)
#ys=np.array([5,6,5,7,8,6,10],dtype=np.float64)


#creating a dataset for testing
def create_dataset(hm,variance,step=2,correlation=False,which_type='pos'):
	val=1
	ys=[]
	for i in range(hm):
		y=val+random.randrange(-variance,variance)
		ys.append(y)
		if correlation==True:
			if which_type=='pos':
				val+=step
			elif which_type=='neg':
				val-=val   
	xs=[i for i in range(len(ys))]			   
	return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)
	
#defining a function to get the best fit slope
#here we pass in the array of points in the function and get the slope anf the intercept 
def best_fit_slope_and_intercept(xs,ys):
	m =(((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2)-mean(xs*xs)))
	b=mean(ys)-m*mean(xs)
	return m,b

# defining function to predict the outcome from the model 
def predict_outcome(x_predict,m,b):
	y=m*x_predict+b
	return y

#defining a function for measuring r-squared
def sum_of_squared_error(ys_original,ys_line):
	return sum((ys_original-ys_line)**2)

def r_squared_value(ys_original,ys_line):
	ys_mean=[mean(ys_original) for y in ys_original]
	squared_error_regression_line=sum_of_squared_error(ys_original,ys_line)
	squared_error_mean_line=sum_of_squared_error(ys_original,ys_mean)
	return (1-(squared_error_regression_line/squared_error_mean_line))

#getting dataset
xs,ys=create_dataset(40,10,5,True,'pos')
	
#getting the slope and the intercept
m,b=best_fit_slope_and_intercept(xs,ys)

#making the regression line list using list comprehension
regression_line=[(m*x+b) for x in xs]

#making prediction using the data 
x_predict=8
y_predict=predict_outcome(x_predict,m,b)

# r-squared and checking whether the best fit line really fits our  data well or not 
#We can have a best fit line for a poorly fit dataset also, the point is to have a metric
#can actually tell us whether th line represents the actual relationship and explains a significant
#percentage of the variation.
r_squared=r_squared_value(ys,regression_line)

#visualise the results
from matplotlib import style
import matplotlib.pyplot as plt
style.use('fivethirtyeight')
plt.scatter(xs,ys,color='red')
plt.plot(xs,regression_line)# this connects the points and gives us a line 
plt.scatter(x_predict,y_predict,s=100,color='green')
plt.show()