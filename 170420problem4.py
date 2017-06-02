# sta 141c python problem3

# cd  C:/Users/toshiya/Desktop/sta141c/hw1_data
# python

######### homework 1 prob4

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



######## make stop words list ###################
training_csv = pd.read_csv('training_csv', header=None) 
training_csv = training_csv[['question1','question2']]


l1 = []
for k in range(len(training_csv)):
	wordlist   = training_csv['question1'][k].split()
	l1 = l1 + wordlist
	

l2 = []
for k in range(len(training_csv)):
	wordlist   = training_csv['question2'][k].split()
	l2 = l2 + wordlist

l = l1+l2


wordfreq = []
for w in l:
    wordfreq.append(l.count(w))


df_words = pd.concat([pd.DataFrame(l),pd.DataFrame(wordfreq)],axis=1)
df_words.columns = ['word','freq']
df_words = df_words[df_words['freq'] > 10000]
del_words = set(df_words['word']) # get unique words




######## delete stop words ############
b1 = []
for q in range(len(df_training_qs)):
	a = df_training_qs['question1'][q].split()
	for w in del_words:
		if w in df_training_qs['question1'][q].split():
			a.remove(w)
	b1 = b1 + [a]


b2 = []
for q in range(len(df_training_qs)):
	a = df_training_qs['question2'][q].split()
	for w in del_words:
		if w in df_training_qs['question2'][q].split():
			a.remove(w)
	b2 = b2 + [a]

df_deled_sent = pd.DataFrame(pd.concat([pd.Series(b1),pd.Series(b2)],axis=1)) # srop words deleted
df_deled_sent.columns = ('question1','question2')

# making new score list 

score_list = []
for k in range(len(df_deled_sent)):
	a   = df_deled_sent['question1'][k]
	b   = df_deled_sent['question2'][k]
	c = 0
	for j in range(len(a)):
		if a[j] in b:
			c += 1 
	for i in range(len(b)):
		if b[i] in a:
			c += 1
	if len(a)+len(b) > 0:
		score = c/(len(a)+len(b))
	else:
		score = 0
	score_list = score_list + list([score])



# calculating accuracy 

df_training['score'] = score_list
df_training['sign']  = df_training['score'] - float(sys.argv[2])

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
