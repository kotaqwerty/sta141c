########## sta141 c hw3 #
# cd  C:/Users/toshiya/Desktop/sta141c/hw3_data
# python

##########
########################################### problem 1 ###############

import sys
import numpy as np
import scipy 
import scipy.sparse.linalg
from scipy import linalg
from scipy.sparse import csr_matrix
import pandas as pd
import timeit
from timeit import Timer 
import time
import sklearn
from sklearn.linear_model import Ridge
from sklearn.datasets import load_svmlight_file


# reading data
def get_data():
    data = load_svmlight_file("cpusmall_scale.txt")
    return data[0], data[1]

X, y = get_data()


# to dataframe


data = pd.concat([pd.DataFrame(y),pd.DataFrame(X.todense())],axis=1)
data.columns = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']


data_training = data.sample(round(len(data) * 0.8))
data_test = data.drop(data_training.index)

X_training = data_training[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']].as_matrix()
y_training = data_training['y'].as_matrix()
y_training = np.matrix(y_training).T


## 1 
alpha = 1
G = np.eye(X_training.shape[1])
params = np.dot(np.linalg.inv(np.dot(X_training.T, X_training) + (alpha * G)),np.dot(X_training.T, y_training))

mse = np.square(((X_training).dot(np.asmatrix(params)) - np.asmatrix(y_training).T)).sum()/len(X_training)



# 2 

def ridge_reg(x,y,alpha):
	G = np.eye(x.shape[1])
	params = np.dot(np.linalg.inv(np.dot(x.T, x) + (alpha * G)),np.dot(x.T, y))
	mse = np.square(((x).dot(np.asmatrix(params)) - np.asmatrix(y).T)).sum()/len(x)
	return params, mse


a = ridge_reg(X_training,y_training,1)
a[1]
a = ridge_reg(X_training,y_training,0.01)
a[1]
a = ridge_reg(X_training,y_training,0.1)
a[1]
a = ridge_reg(X_training,y_training,10)
a[1]
a = ridge_reg(X_training,y_training,100)
a[1]



# ###### 2




def grad_dec_fix_step(x,y,step_size,e):
	w = np.random.rand(x.shape[1],1)
	r = np.linalg.norm((x.T.dot((x).dot(w))) - x.T.dot(y) + w)
	for i in range(50):
		g = ((x.T.dot((x).dot(w))) - x.T.dot(y) + w)
		if np.linalg.norm(g) <= e * r:
			break 
		w = w - step_size * g 
#		print(i)
#		print(w[0])
	return w

grad_dec_fix_step(X_training,y_training,0.0001,0.01)
grad_dec_fix_step(X_training,y_training,0.001,0.01)
grad_dec_fix_step(X_training,y_training,0.01,0.01)
grad_dec_fix_step(X_training,y_training,0.1,0.01)
grad_dec_fix_step(X_training,y_training,0.1,0.01)






########### 4 #########

# reading data
def get_data():
    data = load_svmlight_file("E2006_train.txt")
    return data[0], data[1]

X_train, y_train = get_data()
y_train = np.matrix(y_train).T

grad_dec_fix_step(X_train,y_train,0.001,0.001)
grad_dec_fix_step(X_train,y_train,0.01,0.001)
grad_dec_fix_step(X_train,y_train,0.1,0.001)
grad_dec_fix_step(X_train,y_train,1,0.001)


w = grad_dec_fix_step(X_train,y_train,0.0001,0.001)
mse = np.square(((X_train).dot(w) - (y_train))).sum()/X_train.shape[0]
mse # 0.16
w = grad_dec_fix_step(X_train,y_train,0.001,0.001)
mse = np.square(((X_train).dot(w) - (y_train))).sum()/X_train.shape[0]
mse # 5.15
w = grad_dec_fix_step(X_train,y_train,0.01,0.001)
mse = np.square(((X_train).dot(w) - (y_train))).sum()/X_train.shape[0]
mse # inf



