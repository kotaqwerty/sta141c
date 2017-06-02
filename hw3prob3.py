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



##### ridge reg



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



def ridge_reg(x,y,alpha):
	G = np.eye(x.shape[1])
	params = np.dot(np.linalg.inv(np.dot(x.T, x) + (alpha * G)),np.dot(x.T, y))
	mse = np.square(((x).dot(np.asmatrix(params)) - np.asmatrix(y).T)).sum()/len(x)
	return params, mse

ridge_reg(X_training,y_training,1) # mse = 3718723

clf = Ridge(alpha=1.0)
a = clf.fit(X_training,y_training)
mse = np.square(((X_training).dot(np.asmatrix(a.coef_).T) - np.asmatrix(y_training).T)).sum()/len(X_training) # 23235505
print(mse)





###### logistic reg 

from sklearn import linear_model
import random
import math

def get_data():
    data = load_svmlight_file("news20.txt")
    return data[0], data[1]

X, y = get_data()

l = list(range(y.shape[0]))
l1 = random.sample(l, round(y.shape[0] * 0.8))
l2 = [x for x in l if x not in l1]

X_training = X[l1]
y_training = y[l1]
y_training = np.matrix(y_training).T


X_test = X[l2]
y_test = y[l2]
y_test = np.matrix(y_test).T


ols = linear_model.LogisticRegression()
res = ols.fit(X_training, y_training) # takes 5 mins 



def score_cal(predicted):
	sign_list = []
	for h in range(len(predicted)):
		if predicted[h] > 0:
			sign_list.append(1)
		else:
			sign_list.append(-1)
	d = 0
	for i in range(len(predicted)):
		if sign_list[i] == int(y_test[i]):
			d += 1
	score_acc = (d/len(predicted))   # calculate accuracy
	print(score_acc)                   # return the result


w = res.coef_
pred = X_test.dot(w.T)
score_cal(pred) # 0.92