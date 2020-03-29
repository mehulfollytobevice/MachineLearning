# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:29:09 2020

@author: Mehul
"""
# Thompson sampling
'''
In the multi-arm bandit problem we have to explore and exploit the bandits,
So that we can figure out which machine has the highest return while playing 
on the machines.The past decisions affect the future choices of the machines. 
''' 

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

# using random selection
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

#implmenting Thompson Sampling
import random
'''Step 1:Initialising all the variables '''
N=10000
d=10
number_of_rewards_0=[0]*d
number_of_rewards_1=[0]*d
ads_selected=[]
total_reward=0
'''Step 2:Making the strategy to select the ads at each round'''
for n in range(0,N):
	ad=0
	max_random=0
	for i in range(0,d):
		random_beta=random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)
		if random_beta > max_random:
			max_random=random_beta
			ad=i
			
	'''Step 3:Updating the number_of_rewards_1 and the number_of_rewards_0 '''
	ads_selected.append(ad)
	reward=dataset.values[n,ad]
	if reward==1:
		number_of_rewards_1[ad]=number_of_rewards_1[ad]+1
	else:
		number_of_rewards_0[ad]=number_of_rewards_0[ad]+1
	total_reward=total_reward+reward
	
'''
we observe that the total reward using Thompson Sampling
is more than double the total reward obtained in random selection 
'''

#Visualising the experiment
plt.hist(ads_selected)
plt.xlabel("Ads")
plt.ylabel("Number of times selected")
plt.show()

'''
The plot clearly shows the best ad which has the highest conversion rate
'''