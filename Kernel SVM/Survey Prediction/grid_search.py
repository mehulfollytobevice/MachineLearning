# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:59:25 2020

@author: Mehul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Social_Network_Ads.csv")
'''
for X we are using only two factors , 
age and salary to apply kernel SVM and find out who purchases the product and who does not 
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

#Building the classifier
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

#applying k fold cross validation
from sklearn.model_selection import cross_val_score
'''this will return the k accuracies of the k experiments '''
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()
'''
Standard deviation can tell us the range of the model's accuracy
The accuracy varies from range (mean-std,mean+std) and accordingly the model is put in one of the four categories:
	1.Low Bias Low Variance
	2.Low Bias High Variance
	3.High Bias Low Variance
	4.High Bias High Variance
 
'''

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
'''Here each dictionary represents a particular configuration for a model'''
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
			 {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.001,0.0001]}
			 ]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
'''grid search also incorporates k fold cross validation to get accuracy'''
grid_search=grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

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
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
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
plt.title('Kernel SVM(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()