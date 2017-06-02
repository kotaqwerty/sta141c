# sta 141c python problem2

# cd  C:/Users/toshiya/Desktop/sta141c/hw1_data
# python
######### homework 1 prob2

import sys
import numpy as np
import pandas as pd

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
		str_out = pd.concat([str_out,str_out_i],axis=1)
	return str_out

df_training_qs = preprocess(df_training_qs)


# compute score
for k in range(len(df_training_qs)):
	a   = df_training_qs['question1'][k].split()
	b   = df_training_qs['question2'][k].split()
	c = 0
	for j in range(len(a)):
		if a[j] in b:
			c += 1 
	print(('score_{!r}:').format(k) +str((c/(len(a)+len(b)))) ) 

	