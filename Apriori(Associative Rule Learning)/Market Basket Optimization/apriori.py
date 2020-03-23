# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:04:36 2020

@author: Mehul
"""

#this time we are not importing any package to implement the apriori model
#we are using a file from the python software foundation to implement the algorithm

# importing some important  libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)

'''
The apriori model does not want a dataframe as in input , it wants a lists of lists as the input
'''
""" using list comprehension to make a list of lists"""
transactions=[]
for i in range(0,7501):
	transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

"""other method to make list of lists"""
transactions2=[]
for i in range(0,7501):
	l=[]
	for j in range(0,20):
		l.append(str(dataset.values[i,j]))
	transactions2.append(l)
	
#training apriori on the dataset
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

'''
This algorithm works on a trial and error method , we try different values for the support and confidence until the rules make sense.
We also apply rule for some period of time to see how they work out.

Explanation for all the values chosen-

min_support :
	We want products we are bought atleast 3 times a day, since the dataset contains all purchases for a week.
	The total purchases are 21 in a week , so min_support = 21/7501
	
min_confidence:
	High confidence values will not give any tangible and observable results.
	First , value taken is .8 ;which means that the rule has to be right 80% of the time.
	But this gives only obvious rules , or gives products which are not associated but are sold most overall.
	So then value of min_confidence is decreased up until .2 , when some good associations are seen.

min_lift: 3 will be a good starting value

min_length: We want alteast 2 products in a rule ,since having single products in a rule does not make sense.

'''

#Visualising the rules
results=list(rules)
print(results)
df=pd.DataFrame(results,columns=['rules','support','Stats'])
''' manipulate the dataset and extract meaningful info'''
	
	
		