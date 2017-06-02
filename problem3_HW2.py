# sta 141c python

# cd  C:/Users/toshiya/Desktop/sta141c/hw1_data
# python
######### homework 2 

import sys
import numpy as np
import pandas as pd
import csv

import scipy.sparse.linalg
from scipy import linalg
from scipy.sparse import csr_matrix
import pandas as pd
import timeit
from timeit import Timer 
import time

# problem 3


######## problem3


## define compute ppmi
def compute_ppmi(data):
	f = open(data,'rt',encoding='latin1')  # utf 8 for all
	reader = csv.reader(f)
	global_dict = {}
	dic = {}
	for line in reader:
		q1 = line[3]
		q2 = line[4]
		q1 = q1.split()
		q2 = q2.split()
		q  = q1 + q2
		for w in q:
			if w not in global_dict:
				global_dict[w] = 1
			else:
				global_dict[w] +=1
	D = len(dic)
	n = len(global_dict)
	# make id for all words
	word_id = pd.DataFrame(list(global_dict.items())).reset_index()
	f = open(data,'rt',encoding='latin1') # test 13sec
	reader = csv.reader(f)
	dic_id = {}
	for line in reader:
		q1 = line[4]
		q2 = line[5]
		q1 = q1.split()
		q1_id = list(pd.DataFrame(q1).merge(word_id,on=0,how ='left')[1])
		q2 = q2.split()
		q2_id = list(pd.DataFrame(q2).merge(word_id,on=0,how ='left')[1])
		for i in range(len(q1_id)-3):
			if (q1_id[i],q1_id[i+3]) not in dic_id:
				dic_id[(q1_id[i],q1_id[i+3])] = 1
			else:
				dic_id[(q1_id[i],q1_id[i+3])] += 1
		for i in range(len(q2_id)-3):
			if (q2_id[i],q2_id[i+3]) not in dic:
				dic_id[(q2_id[i],q2_id[i+3])] = 1
			else:
				dic_id[(q2_id[i],q2_id[i+3])] += 1
	## ppmi csr
	from_id = pd.DataFrame(pd.DataFrame(list(dic_id))[0])
	from_id.columns = ['index']
	from_id = from_id.merge(word_id,how='left',on='index')
	to_id = pd.DataFrame(pd.DataFrame(list(dic_id))[1])
	to_id.columns = ['index']
	to_id = to_id.merge(word_id,how='left',on='index')
	vals = np.log(pd.DataFrame(list(dic_id.values()))[0]/from_id[1]*to_id[1])
	ppmi = csr_matrix((vals,(list(from_id['index']),list(to_id['index']))),shape=(n,n))
	return ppmi


ppmi = compute_ppmi('training.csv')
vals, vecs = scipy.sparse.linalg.eigs(ppmi,k=100)
F = vecs.dot(np.diag(vals))


f = open('training.csv','rt',encoding='latin1') # test 13sec
reader = csv.reader(f)
dic_id = {}
sim_list = []
global_dict = {}
for line in reader:
	q1 = line[3]
	q2 = line[4]
	an = line[5]
	q1 = q1.split()
	q2 = q2.split()
	q = q1 + q2
	for w in q:
		if w not in global_dict:
			global_dict[w] = 1
		else:
			global_dict[w] +=1
	# make id for all words
	word_id = pd.DataFrame(list(global_dict.items())).reset_index()
	q1_id = list(pd.DataFrame(q1).merge(word_id,on=0,how ='left')[1])
	q1_id = [x for x in q1_id if str(x) != 'nan'] # omit nan
	q2_id = list(pd.DataFrame(q2).merge(word_id,on=0,how ='left')[1])
	q2_id = [x for x in q2_id if str(x) != 'nan'] # omit nan
	##
	xq1_1 = (1/len(q1_id))
	xq1_2 = np.zeros(100)
	for k in range(len(q1_id)):
		xq1_2 = xq1_2+F[q1_id[k]]
	#
	xq1 = xq1_1 * xq1_2
	#
	xq2_1 = (1/len(q2_id))
	xq2_2 = np.zeros(100)
	for k in range(len(q2_id)):
		xq2_2 = xq2_2+F[q2_id[k]]
	#
	xq2 = xq2_1 * xq2_2
	cos_sim = xq1.T.dot(xq2)/(np.linalg.norm(xq1)*np.linalg.norm(xq2))
	sim_list = sim_list + list([cos_sim])


df_training = pd.read_csv('test5.csv', header=None,encoding='latin1')
df_training = df_training.dropna()
df_training = df_training.reset_index(drop=True)
df_training.columns = ['id','qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

thr = 1.0
sign = pd.DataFrame(np.array(sim_list) - thr)

sign_list = []
for k in range(len(sign)):
	if sign[0][k] > 0:
		sign_list.append(1)
	else:
		sign_list.append(0)

d = 0
for k in range(len(df_training)):
	if sign_list[k] == df_training['is_duplicate'][k]:
		d +=1

score_acc = d/len(sign)
print(score_acc) 





## for validataion

ppmi = compute_ppmi('validation.csv')


vals, vecs = scipy.sparse.linalg.eigs(ppmi,k=100)
F = vecs.dot(np.diag(vals))


f = open('validation.csv','rt',encoding='latin1') # test 13sec
reader = csv.reader(f)
dic_id = {}
sim_list = []
global_dict = {}
for line in reader:
	q1 = line[3]
	q2 = line[4]
	an = line[5]
	q1 = q1.split()
	q2 = q2.split()
	q = q1 + q2
	for w in q:
		if w not in global_dict:
			global_dict[w] = 1
		else:
			global_dict[w] +=1
	# make id for all words
	word_id = pd.DataFrame(list(global_dict.items())).reset_index()
	q1_id = list(pd.DataFrame(q1).merge(word_id,on=0,how ='left')[1])
	q1_id = [x for x in q1_id if str(x) != 'nan'] # omit nan
	q2_id = list(pd.DataFrame(q2).merge(word_id,on=0,how ='left')[1])
	q2_id = [x for x in q2_id if str(x) != 'nan'] # omit nan
	##
	xq1_1 = (1/len(q1_id))
	xq1_2 = np.zeros(100)
	for k in range(len(q1_id)):
		xq1_2 = xq1_2+F[q1_id[k]]
	#
	xq1 = xq1_1 * xq1_2
	#
	xq2_1 = (1/len(q2_id))
	xq2_2 = np.zeros(100)
	for k in range(len(q2_id)):
		xq2_2 = xq2_2+F[q2_id[k]]
	#
	xq2 = xq2_1 * xq2_2
	cos_sim = xq1.T.dot(xq2)/(np.linalg.norm(xq1)*np.linalg.norm(xq2))
	sim_list = sim_list + list([cos_sim])


df_training = pd.read_csv('validation.csv', header=None,encoding='latin1')
df_training = df_training.dropna()
df_training = df_training.reset_index(drop=True)
df_training.columns = ['id','qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

thr = 1.0
sign = pd.DataFrame(np.array(sim_list) - thr)

sign_list = []
for k in range(len(sign)):
	if sign[0][k] > 0:
		sign_list.append(1)
	else:
		sign_list.append(0)

d = 0
for k in range(len(df_training)):
	if sign_list[k] == df_training['is_duplicate'][k]:
		d +=1

score_acc = d/len(sign)
print(score_acc)  ## 0.6300 for validation acc





