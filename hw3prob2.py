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
import random
import math


# reading data
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




def log_grad_dec_fix_step(x,y,step_size,e):
	w = np.matrix(np.zeros(X_training.shape[1])).T
	r = np.linalg.norm(x.T.dot((-1)*y) * (math.e)**int(-y.T.dot(x.dot(w)))/ (1+(math.e)**int(-y.T.dot(x.dot(w))))  + w)
	for i in range(50):
		g = x.T.dot((-1)*y) * (math.e)**int(-y.T.dot(x.dot(w)))/ (1+(math.e)**int(-y.T.dot(x.dot(w))))  + w
		if np.linalg.norm(g) <= e * r:
			break 
		w = w - step_size * g 
#		print(i)
#		print(w[0])
	return w


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


w = log_grad_dec_fix_step(X_training,y_training,0.01,0.001)
pred = X_test.dot(w)
score_cal(pred) # 0.777
