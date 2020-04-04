# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:16:47 2020

@author: Mehul
"""
#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset=pd.read_csv("train_formatted.csv")

	
X=dataset.iloc[:,[1,3,4,5,6,7,8,9,10]].values
y=dataset.iloc[:,0].values

#dealing with missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values=np.nan,strategy='mean',axis=0)
imputer=imputer.fit(X[:,2:3])
X[:,2:3]=imputer.transform(X[:,2:3])

#categorical encoding 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,7]=labelencoder_X.fit_transform(X[:,7])
X[:,8]=labelencoder_X.fit_transform(X[:,8])

#Removing ticket column
X=X[:,[0,1,2,3,4,6,7,8]]

#Dummy coding using OneHotEncoder
#Removing one variable to avoid dummy variable trap
#X
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
onehotencoder=OneHotEncoder(categorical_features=[7])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
onehotencoder=OneHotEncoder(categorical_features=[15])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_X2=StandardScaler()
# scaling dummy variables
X_train=sc_X2.fit_transform(X_train)
X_test=sc_X2.transform(X_test)

#Applying KNN model

#building the classifier
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,p=2,metric="minkowski")
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

#applying k-fold cross validation
from sklearn.model_selection import cross_val_score
'''this will return the k accuracies of the k experiments '''
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
knn_accuracy=accuracies.mean()
accuracies.std()

#Appying Kernel SVM
#Buiding the classifier
from sklearn.svm import SVC
#we can change the type of kernel 
classifier=SVC(kernel='rbf',random_state=0)
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

#applying k-fold cross validation
from sklearn.model_selection import cross_val_score
'''this will return the k accuracies of the k experiments '''
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
kernelsvm_accuracy=accuracies.mean()
accuracies.std()

#applying grid search
from sklearn.model_selection import GridSearchCV
'''Here each dictionary represents a particular configuration for a model'''
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
			 {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.01,0.001,0.002,0.003,0.004,0.005]}
			 ]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
'''grid search also incorporates k fold cross validation to get accuracy'''
grid_search=grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#Applying Random Forest 
#Building the classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=5000,criterion="entropy",random_state=0)
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

#applying k-fold cross validation
from sklearn.model_selection import cross_val_score
'''this will return the k accuracies of the k experiments '''
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
random_accuracy=accuracies.mean()
accuracies.std()

#So far we get the best accuracy from Kernel SVM and random forest 
#We can apply grid search to optimise the model and increase the accuracy
