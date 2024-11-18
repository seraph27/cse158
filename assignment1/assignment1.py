# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
def Jaccard(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

# %%
allbooks = set()
for user in ratingsPerUser:
    for book, _ in ratingsPerUser[user]:
        allbooks.add(book)

# %%
newRatingsValid = [(user, book, 1) for user, book, _ in ratingsValid]
for user, book, _ in ratingsValid:
    newBook = random.choice(list(allbooks))
    while newBook in ratingsPerUser[user]:
        newBook = random.choice(list(allbooks))
    newRatingsValid.append((user, newBook, 0))

# %%
def solve(threshold1, threshold2):
    ans1 = set()
    for user, book, _ in ratingsValid:
        B = ratingsPerItem[book]
        maxSim = 0
        for bprime, _ in ratingsPerUser[user]:
            if bprime == book:
                continue
            B2 = ratingsPerItem[bprime]
            sim = Jaccard(set([x[0] for x in B]), set([x[0] for x in B2]))
            maxSim = max(maxSim, sim)
            #print(maxSim)
        if maxSim > threshold1:
            ans1.add(book)

    bookCount = defaultdict(int)
    totalRead = 0

    for user,book,_ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()
    count = 0
    ans2 = set()
    for ic, i in mostPopular:
        count += ic
        ans2.add(i)
        if count > totalRead * threshold2: break

    global ret
    ret = set.union(ans1, ans2)
    correct = 0
    for _, book, predict in newRatingsValid:
        correct += predict == (book in ret)
    
    acc = correct / len(newRatingsValid)
    return acc

#took this from assignment 3
    


# %%
solve(0.001, 0.5)

# %%
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    if b in ret:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")


predictions.close()


