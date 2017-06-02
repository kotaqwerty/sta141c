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
# reading data

#f_data = pd.read_csv('training.csv', header=None)
#f_data.head(50000).to_csv('test5.csv')


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


## problem2 with preprocess

# define preprocess
char_list = "?,!().';/"

def preprocess(str_in):
	s = str_in.lower()
	for c in char_list:
		if c in s:
			s = s.replace(c,' ')
	return s.replace('-', '')

# define compute_ppmi with preprocess
def compute_ppmi_preprocess(data):
	f = open(data,'rt',encoding='latin1')  # utf 8 for all
	reader = csv.reader(f)
	global_dict = {}
	dic = {}
	for line in reader:
		q1 = preprocess(line[3])
		q2 = preprocess(line[4])
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
		q1 = preprocess(line[4])
		q2 = preprocess(line[5])
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

ppmi_pre = compute_ppmi_preprocess('training.csv')
scipy.sparse.linalg.norm(ppmi_pre,ord='fro') # 1629.29













