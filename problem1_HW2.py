########## sta141 c hw2 #
# cd  C:/Users/toshiya/Desktop/sta141c/hw2_data
# python

##################################################### problem 1 ###############

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
# reading data
links = pd.read_table("enwiki-2013.txt.gz", sep = " ",
        dtype = np.int32, skiprows = 4, engine = "c",
        names = ("from", "to")
        )

# count unique id 
id_list = links['from'].append(links['to'])
n = len(id_list.unique())

# make sprs matrix
vals = np.ones(len(links))
A1 = csr_matrix((vals, (links['from'], links['to'])),shape = (4206289,4206289))


# power method
def power_method(sprs,iterations):
	d = sprs.shape[1]
	v = np.random.rand(d,1)
	for k in range(iterations):
		v = sprs.dot(v) 
		v = v/(np.linalg.norm(v))
	eigen_val = v.T.dot(sprs.dot(v))
	return v, eigen_val


# comparing calculating time 
if __name__ == '__main__':
    sprs,iterations = A1, 1

timeit.timeit('power_method(sprs,iterations)',setup='from __main__ import power_method,sprs,iterations',number=1) # 1.29

if __name__ == '__main__':
    sprs,iterations = A1, 3

timeit.timeit('power_method(sprs,iterations)',setup='from __main__ import power_method,sprs,iterations',number=1) # 2.87

if __name__ == '__main__':
    sprs,iterations = A1, 5

timeit.timeit('power_method(sprs,iterations)',setup='from __main__ import power_method,sprs,iterations',number=1) # 4.43

if __name__ == '__main__':
    sprs,iterations = A1, 20

timeit.timeit('power_method(sprs,iterations)',setup='from __main__ import power_method,sprs,iterations',number=1) # 16.57



time_start = time.clock()
scipy.sparse.linalg.svds(A1,k=1)
time_elapsed = (time.clock() - time_start) # 32.56
print(time_elapsed)

time_start = time.clock()
scipy.sparse.linalg.svds(A1,k=3)
time_elapsed = (time.clock() - time_start)
print(time_elapsed) # 38.67

time_start = time.clock()
scipy.sparse.linalg.svds(A1,k=5)
time_elapsed = (time.clock() - time_start)
print(time_elapsed) # 48.91

time_start = time.clock()
scipy.sparse.linalg.svds(A1,k=20)
time_elapsed = (time.clock() - time_start)
print(time_elapsed) # 122.79

quality = ret[0].T.dot(A1.T).dot(A1.dot(ret[0]))/scipy.sparse.linalg.norm(A1,ord='fro')


# test
u ,s ,v = scipy.sparse.linalg.svds(A1,k=1)





## list top 5  authoruty

result = scipy.sparse.linalg.svds(A1,k=1)

df_right = pd.DataFrame(result[2].T)
df_right_sort = df_right.sort(0,ascending=False)

# 2038044 : United States
# 3165770 : List of sovereign states
# 186871  : Geographic Names Information System
# 2774619 : Political divisions of the United States
# 186873  : Federal Information Processing Standard

## list top 5 hub


df_left = pd.DataFrame(result[0])
df_left_sort = df_left.sort(0,ascending=False)

# 195374 Cleveland
# 210477 list of primate cities
# 192859 List of University of Pennsylvania people
# 1186296 History of Western civilization
# 4030617 Central sulcus



## pagerank
n = A1.shape[0]

# make marcov matrix

csums = np.array(A1.sum(0))[0,:]
ri,ci= A1.nonzero()
A1.data /= csums[[ci]]


# calculate pageranks
x = np.ones(n)/n
e = np.matrix(np.ones(n)).T

for k in range(3):
	x = (0.9*A1.T).dot(x) + ((0.1/n)*(e.dot(e.T))).dot(x)






for k in range(3):
	x = (0.9*A1.T).dot(x) + ((0.1/n)*(e.dot(e.T)))


def pagerank(sprs,k):
	n = A1.shape[0]




# test

from_id =   (1, 1, 1, 1, 1, 2, 4)
to_id =     (0, 2, 3, 4, 5, 3, 3)
vals = np.ones(len(from_id))
A2 = csr_matrix((vals, (from_id, to_id)), shape=(6, 6))

n = A2.shape[0]

csums = np.array(A2.sum(0))[0,:]
ri,ci= A2.nonzero()
P = A2.data/csums[[ci]]
A2.data /=  csums[[ci]]

rsums = np.array(A2.sum(1))[:,0]
ri, ci = A2.nonzero()
A2.data /= rsums[ri]