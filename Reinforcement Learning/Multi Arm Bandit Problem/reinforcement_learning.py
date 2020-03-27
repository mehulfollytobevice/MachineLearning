# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:10:02 2020

@author: Mehul
"""
#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
Note:In re-enforcement learning we start with no data,
and all the decisions are based on the data we have encountered so far.
That is why this is called online learning or interactive learning.
It is very useful in Artificial Intelligence.
The dataset given here is just for simulation,
in reality we build the dataset as we train the machine.
'''

#import the dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#implementing random selection
import random
total_reward=0
types=10
N=10000
choices=[]
for i in range(0,N):
	n=random.randrange(0,types)
	choices.append(n)
	reward=dataset.values[i,n]
	total_reward=total_reward +reward


#plotting a histogram
plt.hist(choices)
plt.xlabel("Ads")
plt.ylabel("Number of times selected")
plt.show()
'''
from the plot we can see that we get an even distribution when ads are randomly selected
And we observe that the total reward is around 1200
'''

#implmenting UCB
import math
'''Step 1:Initaialising all the variables '''
N=10000
d=10
number_of_selections=[0]*d
sum_of_rewards=[0]*d
ads_selected=[]
total_reward=0
'''Step 2:Making the strategy to select the ads at each round'''
for n in range(0,N):
	ad=0
	max_upper_bound=0
	for i in range(0,d):
		if (number_of_selections[i]>0):
			''' Condition is applied to take care of when n<d'''
			average_reward=sum_of_rewards[i]/number_of_selections[i]
			delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selections[i])
			upper_bound=average_reward+delta_i
		else:
			'''Huge Upper Bound Trick: To ensure that each ad is selected in the intial steps '''
			upper_bound=1e400#any riduculously huge number would do
		
		if upper_bound > max_upper_bound:
			max_upper_bound =upper_bound
			ad=i
			
	'''Step 3:Updating the ads_selected vector and the sum_of_rewards vector '''
	ads_selected.append(ad)
	number_of_selections[ad]=number_of_selections[ad]+1
	reward=dataset.values[n,ad]
	sum_of_rewards[ad]=sum_of_rewards[ad]+reward
	total_reward=total_reward+reward
	
'''
we observe that the total reward using UCB
is almost double than the total reward obtained in random selection 
'''

#Visualising the experiment
plt.hist(ads_selected)
plt.xlabel("Ads")
plt.ylabel("Number of times selected")
plt.show()

'''
The plot clearly shows the best ad which has the highest conversion rate
Even the sum_of_rewards vector shows the ad having the highest reward 
'''