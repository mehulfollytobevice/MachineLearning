# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 03:42:37 2020

@author: Mehul
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#importing the dataset
dataset=pd.read_csv('OnlineNewsPopularity.csv')

#deleting unecessary columns from the dataset
dataset=dataset.drop(columns=['url'])

#extracting the matrix of features and the dependent variable 
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
j=0
for i in y:
	if(i>1400):
		y[j]=1
	else:
		y[j]=0
	j+=1
	
#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X2=StandardScaler()
X_train2=sc_X2.fit_transform(X_train)
X_test2=sc_X2.transform(X_test)

#building the classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=500,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

#making the predictions:
y_pred=classifier.predict(X_test)

#making a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
pred_true=cm[0,0]+cm[1,1]
pred_false=cm[0,1]+cm[1,0]
print("correct predictions:"+str(pred_true))
print("incorrect predictions:"+str(pred_false))

#using k fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)
random_accuracy=accuracies.mean()
accuracies.std()



