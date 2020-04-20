# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:22:32 2020

@author: Mehul
"""

#importing the libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

#defining a support vector machine class
class Support_Vector_Machine:
	#making the constructor
	def __init__(self,visualisation=True):
		self.visualisation=visualisation
		self.colors={1:'r',-1:'b'}
		if self.visualisation:
			self.fig=plt.figure()
			self.ax=self.fig.add_subplot(1,1,1)
	
	#training the support vector machine
	def fit(self,data):
		self.data=data
		opt_dict={}
		#these transforms are used to minimise the value of mod of w
		#so that the width of the street can be maximised 
		transforms=[[1,1],[-1,1],[-1,-1],[1,-1]]
		all_data=[]
		for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
		#minimum and maximum value of the features
		self.max_feature_value=max(all_data)
		self.min_feature_value=min(all_data)
		
		all_data=None  #we have to leave some space for processing 
		#point of expense 
		#step sizes are used for optimising the value of w
		step_sizes=[self.max_feature_value*0.1,
			  self.max_feature_value*0.01,
			  self.max_feature_value*0.001]
		
		#apart from w , we need to maximise b also
		b_range_multiple=5 # expensive
		b_multiple=5
		
		latest_optimum=self.max_feature_value*10
		
		#beginning stepping
		for step in step_sizes:
			w=np.array([latest_optimum,latest_optimum])
			#this is where we do the iterations to optimize the w 
			optimized=False
			while not optimized:
				for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
					   (self.max_feature_value*b_range_multiple),step*b_multiple):
					for transformation in transforms:
					
						w_t=w*transformation
						found_option=True
						#weakest point of the SVM
						for i in self.data:
							for xi in self.data[i]:
								yi=i
								if not yi*(np.dot(w_t,xi)+b)>=1:
									found_option=False
						if found_option:
							opt_dict[np.linalg.norm(w_t)]=[w_t,b]
				if w[0]<0:
					optimized=True
					print('Optimized a step')
				else:
					w=w-step
			norms=sorted([n for n in opt_dict])
			opt_choice=opt_dict[norms[0]]
			self.w=opt_choice[0]
			self.b=opt_choice[1]
			latest_optimum=opt_choice[0][0]+step*2
		       
	
	def predict(self,features):
		#classification is done on the basis of the sign of (w.x+b)
		classification=np.sign(np.dot(np.array(features),self.w) +self.b)
		if classification !=0 and self.visualisation:
			self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
		return classification
	
	def visualize(self):
		[[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i])for x in data_dict[i]] for i in data_dict]
		
		def hyperplane(x,w,b,v):
			return (-w[0]*x-b+v)/w[1]
		datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
		hyp_x_min=datarange[0]
		hyp_x_max=datarange[1]
		
		#positive support vector
		psv1=hyperplane(hyp_x_min,self.w,self.b,1)
		psv2=hyperplane(hyp_x_max,self.w,self.b,1)
		self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k')
			
		#negative support vector
		nsv1=hyperplane(hyp_x_min,self.w,self.b,-1)
		nsv2=hyperplane(hyp_x_max,self.w,self.b,-1)
		self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k')
		
		#decision boundary
		db1=hyperplane(hyp_x_min,self.w,self.b,0)
		db2=hyperplane(hyp_x_max,self.w,self.b,0)
		self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')
		
		plt.show()
		 
		
#data dictionary is a ilst of list
data_dict={-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}

#svm 
svm=Support_Vector_Machine()
svm.fit(data=data_dict)

# prediction list
prediction_list=[[3,4],[5,5],[6,5],[2,3],[4,4],[7,5]]# list of the points you want to classify
for pred in prediction_list:
	svm.predict(pred)

svm.visualize()