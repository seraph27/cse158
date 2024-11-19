# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from skopt import gp_minimize

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
bookPerUser = defaultdict(list)
userPerBook = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    bookPerUser[u].append(b)
    userPerBook[b].append(u)

# %%
def Jaccard(s1, s2):
    inter = len(set(s1).intersection(set(s2)))
    union = len(set(s1).union(set(s2)))
    if union == 0:
        return 0
    return inter / union

# %%
allbooks = set()
for _, book, _ in allRatings:
    allbooks.add(book)
allbooks = list(allbooks)
readValidBinary = set()
notReadValidBinary = set()

for user, book, _ in ratingsValid:
    readValidBinary.add((user, book))

for user, book, _ in ratingsValid:
    newBook = random.choice(allbooks)
    while (user, newBook) in ratingsValid or (user, newBook) in notReadValidBinary:
        newBook = random.choice(allbooks)
    notReadValidBinary.add((user, newBook))

ratingsValidBinary = list()
ratingsValidBinary.extend((user, book, 1) for user, book in readValidBinary)
ratingsValidBinary.extend((user, book, 0) for user, book in notReadValidBinary)


# %%
def solve(t1, t2):
    correct = 0
    ret = set()
    for user, book, label in ratingsValidBinary:
        users_for_book = set(ratingsPerItem[book])
        max_jaccard = 0
        for bprime, _ in ratingsPerUser[user]:
            jaccard = Jaccard(set(ratingsPerItem[bprime]), users_for_book)
            max_jaccard = max(max_jaccard, jaccard)
        
        ok = 0
        if max_jaccard > t1: ok = 1
        if len(ratingsPerItem[book]) > t2:
            ok = 1
            # if len(ratingsPerItem[book]) > 150 and max_jaccard == 0:
            #     ok = 0
        
        if ok==label:
            correct+=1
            ret.add((user, book))


    acc = correct / len(ratingsValidBinary)


    return acc, ret


# %%
acc, ret= solve(0.0135, 32)


# %%
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    users_for_book = set(ratingsPerItem[b])
    max_jaccard = 0
    for bprime, _ in ratingsPerUser[u]:
        jaccard = Jaccard(set(ratingsPerItem[bprime]), users_for_book)
        max_jaccard = max(max_jaccard, jaccard)
    
    ok = 0
    if max_jaccard > 0.0135: ok = 1
    if len(ratingsPerItem[b]) > 33:
        ok = 1
    predictions.write(u + ',' + b + ',' + str(ok) + '\n')
predictions.close()

# %% [markdown]
# # BELOW IS RATING PREDICTION

# %%
alpha = np.mean([r for _, _, r in ratingsTrain]) #referenced and edited from hw3 solution. 
beta_user = defaultdict(float)
beta_book = defaultdict(float)
for u in ratingsPerUser:
    beta_user[u] = 0

for b in ratingsPerItem:
    beta_book[b] = 0

def solve(L1, L2):
    alpha = sum(rating - (beta_user[user] + beta_book[book]) for user, book, rating in ratingsTrain) / len(ratingsTrain)
    for user, items in ratingsPerUser.items():
        beta_user[user] = sum(rating - (alpha + beta_book[book]) for book, rating in items) / (L1 + len(items))
    for book, items in ratingsPerItem.items():
        beta_book[book] = sum(rating - (alpha + beta_user[user]) for user, rating in items) / (L2 + len(items))

    validMSE = 0
    for u,b,r in ratingsTrain:
        prediction = alpha + beta_user[u] + beta_book[b]
        validMSE += (r - prediction)**2
    reg_user = 0
    reg_book = 0
    for user in beta_user:
        reg_user += beta_user[user] ** 2
    for book in beta_book:
        reg_book += beta_book[book] ** 2
    return (validMSE, validMSE + L1 * reg_user + L2 * reg_book)


# %%
def solve2(L1, L2): #referenced and edited from hw3 solution. 
    mse, minimize = solve(L1, L2)
    new_mse, new_minimize = solve(L1, L2)
    iter = 2
    while iter < 10 or minimize - new_minimize > 0.0001:
        mse, minimize = new_mse, new_minimize
        new_mse, new_minimize = solve(L1, L2)
        iter+=1

    validMSE = 0
    for u,b,r in ratingsValid:
        bu = 0
        bi = 0
        if u in beta_user:
            bu = beta_user[u]
        if b in beta_book:
            bi = beta_book[b]
        prediction = alpha + bu + bi
        validMSE += (r - prediction)**2

    validMSE /= len(ratingsValid)
    print("Validation MSE = " + str(validMSE))
    return validMSE
    


# %%
print(solve2(3.8, 20.5))


# %%
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    prediction = alpha + beta_user.get(u, 0) + beta_book.get(b, 0)
    predictions.write(u + ',' + b + ',' + str(prediction) + '\n')
    
predictions.close()


