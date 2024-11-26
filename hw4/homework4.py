# %%
import gzip
import math
import numpy
import random
import sklearn
import string
from collections import defaultdict
from nltk.stem.porter import *
from sklearn import linear_model
from gensim.models import Word2Vec
import dateutil
from scipy.sparse import lil_matrix # To build sparse feature matrices, if you like

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1

# %%
dataset = []

f = gzip.open("steam_category.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
Ntrain = 10000
Ntest = 10000

dataTrain = dataset[:Ntrain]
dataTest = dataset[Ntrain:Ntrain + Ntest]

# %%
sp = set(string.punctuation)

# %%


# %%
cntWord = defaultdict(int) #referenced from 11/14 lecture 
totalWords = 0
for d in dataTrain:
    r = d['text'].lower()
    r = ''.join([c for c in r if c not in sp])
    ws = r.split()
    for w in ws:
        cntWord[w] += 1
        totalWords += 1

totalWords

# %%
len(cntWord)

# %%
counts = [(v, k) for k, v in cntWord.items()]
counts.sort(reverse=True)


# %%
counts

# %%
answers['Q1'] = counts[:10]

# %%
assertFloatList([x[0] for x in answers['Q1']], 10)

# %%
### Question 2

# %%
NW = 1000 # dictionary size

# %%
words = [x[1] for x in counts[:NW]]

# %%
words

# %%
wordID = dict(zip(words, range(NW)))
wordSet = set(words)

# %%
def feature(datum): #referenced from 11/14 lecture 
    pos = [0] * NW
    r = datum['text'].lower()
    r = ''.join([c for c in r if c not in sp])
    ws = r.split()
    for w in ws:
        if w in wordSet:
            pos[wordID[w]] += 1

    return pos + [1]

# %%
# Build X...

X = [feature(d) for d in dataset]
y = [d['genreID'] for d in dataset]

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:Ntrain + Ntest]
ytest = y[Ntrain:Ntrain + Ntest]

# %%
mod = linear_model.LogisticRegression(C=1)

# %%
mod.fit(Xtrain, ytrain)
correct = list()
for i in range(Ntest):
    if mod.predict([Xtest[i]]) == ytest[i]:
        correct.append(1)
    else:
        correct.append(0)

# %%
answers['Q2'] = sum(correct) / len(correct)

# %%
assertFloat(answers['Q2'])

# %%
### Question 3

# %%


# %%
answers['Q3'] = 

# %%
assertFloatList([x[0] for x in answers['Q3']], 5)
assertFloatList([x[1] for x in answers['Q3']], 5)

# %%
### Question 4

# %%
# Build X and y...

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]

# %%
mod = linear_model.LogisticRegression(C=1)

# %%


# %%
answers['Q4'] = sum(correct) / len(correct)

# %%
assertFloat(answers['Q4'])

# %%
### Question 5

# %%
def Cosine(x1,x2):
    # ...

# %%


# %%
similarities.sort(reverse=True)

# %%
answers['Q5'] = similarities[0]

# %%
assertFloat(answers['Q5'][0])

# %%
### Question 6

# %%


# %%
answers['Q6'] = 

# %%
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
import dateutil.parser

# %%
dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    d['datetime'] = dateutil.parser.parse(d['date_added'])
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%


# %%
model5 = Word2Vec(reviewLists,
                  min_count=1, # Words/items with fewer instances are discarded
                  vector_size=5, # Model dimensionality
                  window=3, # Window size
                  sg=1) # Skip-gram model

# %%


# %%
answers['Q7'] = res[:5]

# %%
assertFloatList([x[1] for x in answers['Q7']], 5)

# %%
f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



