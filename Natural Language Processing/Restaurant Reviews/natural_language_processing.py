# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:11:41 2020

@author: Mehul
"""
#Natural Language Processing 

#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
'''
The  dataset is of .tsv extension (tab seperated values).
We use this because tabs are uncommon in reviews while commas are not .
Also we ignore the double quotes to avoid any problems .
'''
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#Cleaning the dataset
import re
''' 
Step 1:
We will remove punctuation marks and  numbers .
We apply procedures on the first review , then we iterate it over the dataset.
^ ----> means not . We remove all characters that are not characters from a-z,A-Z
The removed chracter is replaced by a space .
'''
review=re.sub('[^a-zA-Z]'," ",dataset["Review"][0])

'''
Step 2:
Changing the review to lower case.
'''
review=review.lower()

'''
Step 3:
Removing insignificant words like prepositions ,conjunctions , articles.
Basically words that are nor adjectives or nouns
'''
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
review=review.split()
review=[word for word in review if not word in set(stopwords.words("english"))] #list comprehension 

'''
Step 4:
Using stemming to keep only root words , so that we can decrease the amount of essential words.
We keep only the root word , so we can limit the sparsity in the future sparse matrix.
Because 'love','loved','will love' all depict the sam feeling for the machine.
'''
from nltk.stem import PorterStemmer
ps=PorterStemmer()
review=[ps.stem(word) for word in review]
'''
Step 5:
Converting list back to string.
'''
review =' '.join(review)

''' Iterating the process for all of the dataset.Creating a corpus.'''

corpus=[]
for i in range(0,1000):
	review=re.sub('[^a-zA-Z]'," ",dataset["Review"][i])
	review=review.lower()
	review=review.split()
	review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
	review=' '.join(review)
	corpus.append(review)
	
#Creating Bag of Words model through tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray() #the sparse matrix is basically the matrix of features for the classification model
#we can use the max_features parameter to limit the number of words 
#another way of reducing sparsity is dimensionality reduction
y=dataset.iloc[:,[1]].values

#Using Naive Bayes Classification Model

#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Building the classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
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
TN=cm[0,0]
TP=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
'''
Where TN= true negatives and TP= true positives 
Metrics used to see the performance of the models:
1.Accuracy = (TP + TN) / (TP + TN + FP + FN)=.73
2.Precision = TP / (TP + FP)=.6842
3.Recall = TP / (TP + FN)=.8834
4.F1 Score = 2 * Precision * Recall / (Precision + Recall)=.77118
'''

#Using Decision Tree Classification Model

#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

"this model uses splits to classify data , so it basically makes if conditions  "
#Building the classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
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
TN=cm[0,0]
TP=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
'''
Where TN= true negatives and TP= true positives 
Metrics used to see the performance of the models:
1.Accuracy = (TP + TN) / (TP + TN + FP + FN)=.71
2.Precision = TP / (TP + FP)=.74725
3.Recall = TP / (TP + FN)=.660194
4.F1 Score = 2 * Precision * Recall / (Precision + Recall)=.7010309
'''

#Using Random Forest Classification
#splitting the dataset into a training set and a test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#Building the classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=15,criterion="entropy",random_state=0)
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
TN=cm[0,0]
TP=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
'''
Where TN= true negatives and TP= true positives 
Metrics used to see the performance of the models:
1.Accuracy = (TP + TN) / (TP + TN + FP + FN)=.72
2.Precision = TP / (TP + FP)=.82191
3.Recall = TP / (TP + FN)=.5825
4.F1 Score = 2 * Precision * Recall / (Precision + Recall)=.681818 
'''
