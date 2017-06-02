# sta 141c python problem3

# cd  C:/Users/toshiya/Desktop/sta141c/hw1_data
# python

######### homework 1 prob3

import sys
import numpy as np
import pandas as pd
import pickle
# reading data using sys



df_training = pd.read_csv(sys.argv[1], header=None)
df_training = df_training.dropna()
df_training = df_training.reset_index(drop=True)
df_training.columns = ['id','qid1', 'qid2', 'question1', 'question2', 'is_duplicate']

df_training_qs = df_training[['question1','question2']]


# preprocessing data
def preprocess( str_in ):
	numcols = len(str_in.columns)
	str_out = pd.DataFrame()
	for i in range(numcols):
		str_out_i = pd.Series(str_in.iloc[:,i]).str.lower()
		str_out_i = pd.Series(str_in.iloc[:,i]).str.replace('?'," ")
		str_out_i = pd.Series(str_out_i).str.replace('!'," ")
		str_out_i = pd.Series(str_out_i).str.replace(':'," ")
		str_out_i = pd.Series(str_out_i).str.replace(','," ")
		str_out_i = pd.Series(str_out_i).str.replace('.'," ")
		str_out_i = pd.Series(str_out_i).str.replace('('," ")
		str_out_i = pd.Series(str_out_i).str.replace(')'," ")
		str_out_i = pd.Series(str_out_i).str.replace('’'," ")
		str_out_i = pd.Series(str_out_i).str.replace('"'," ")
		str_out_i = pd.Series(str_out_i).str.replace("'"," ")
		str_out_i = pd.Series(str_out_i).str.replace("-","")
		str_out_i = pd.Series(str_out_i).str.lower()
		str_out = pd.concat([str_out,str_out_i],axis=1)
	return str_out

df_training_qs = preprocess(df_training_qs)


# compute score

score_list = []
for k in range(len(df_training_qs)):
	a   = df_training_qs['question1'][k].split()
	b   = df_training_qs['question2'][k].split()
	c = 0
	for j in range(len(a)):
		if a[j] in b:
			c += 1 
	for i in range(len(b)):
		if b[i] in a:
			c += 1
	score = c/(len(a)+len(b))
	score_list = score_list + list([score]) # make score_list

### problem 3


# compute accuracy with thrsh
df_training['score']   = score_list
df_training['sign'] = df_training['score'] - float(sys.argv[2])

sign_list = []
for h in range(len(df_training)):
	if df_training['sign'][h] > 0:
		sign_list.append(1)
	else:
		sign_list.append(0)

df_training['sign_list'] = sign_list

d = 0
for i in range(len(df_training)):
	if df_training['sign_list'][i] == df_training['is_duplicate'][i]:
		d += 1

score_acc = (d/len(df_training))   # calculate accuracy
print(score_acc)                   # return the result
