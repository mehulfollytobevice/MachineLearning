# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:49:19 2020

@author: Mehul
"""
#importing some important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#encoding the dataset
from mlxtend.preprocessing import TransactionEncoder
encoder=TransactionEncoder()
encoder_array=encoder.fit_transform(transactions)
encoded_dataset=pd.DataFrame(encoder_array,columns=encoder.columns_)

#now making the eclat model
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets=apriori(encoded_dataset,min_support=0.005,use_colnames=True)
'''when only support is used then the model becomes eclat model'''
rules=association_rules(df_sets,metric="support",min_threshold=0.005,support_only=True)

''' eclat is just a faster version of the apriori algorithm'''
