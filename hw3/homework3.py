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
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

# %%
##################################################
# Read prediction                                #
##################################################

# %%
ratingsValid[:10]

# %%
# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

# %%
### Question 1

# %%
newRatingsValid = [(user, book, 1) for user, book, _ in ratingsValid]
for user, book, _ in ratingsValid:
    newBook = random.choice(list(bookCount.keys()))
    while newBook in [b for b, _ in ratingsPerUser[user]]:
        newBook = random.choice(list(bookCount.keys()))
    newRatingsValid.append((user, newBook, 0))

# %%
random.shuffle(newRatingsValid)
newRatingsValid[-10:]

# %%
correct = 0
for _, book, predict in newRatingsValid:
    correct += predict == (book in return1)

acc1 = correct / len(newRatingsValid)
print(acc1)

# %%
answers['Q1'] = acc1

# %%
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
def solve(threshold):
    bookCount = defaultdict(int)
    totalRead = 0

    for _ ,book,_ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    return2 = set()
    count = 0

    for ic, i in mostPopular:
        count += ic
        return2.add(i)
        if count > totalRead * threshold: break

    correct = 0
    for _, book, predict in newRatingsValid:
        correct += predict == (book in return2)
    #print(correct, return2)
    acc = correct / len(newRatingsValid)
    return acc

# %%
l, r, eps = 0, 1, 1e-3
while r - l > eps:
    m1 = l + (r - l) / 3
    m2 = r - (r - l) / 3
    print(m1, m2)
    if solve(m1) < solve(m2):
        l = m1
    else:
        r = m2

best_threshold = l



# %%
acc2 = solve(best_threshold)
acc2

# %%
answers['Q2'] = [best_threshold, acc2]

# %%
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

# %%
### Question 3/4

# %%
def Jaccard(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2)) 

# %%
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
for user, item, _ in ratingsTrain:
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

def solveQ3(threshold):
    ans = set()
    for user, book, _ in newRatingsValid:
        B = usersPerItem[book] #all users who read the book
        maxSim = 0
        for bprime in itemsPerUser[user]:
            if bprime == book:
                continue
            B2 = usersPerItem[bprime]
            sim = Jaccard(B, B2)
            maxSim = max(maxSim, sim)
            #print(maxSim)
        if maxSim > threshold:
            ans.add(book)

    correct = 0
    for _, book, predict in newRatingsValid:
        correct += predict == (book in ans)
    
    acc = correct / len(newRatingsValid)
    return acc


# %%
l, r, eps = 0.006, 0.009, 1e-6
while r - l > eps:
    m1 = l + (r - l) / 3
    m2 = r - (r - l) / 3
    now1 = solveQ3(m1)
    now2 = solveQ3(m2)
    print(now1, l, now2, r)
    if now1 < now2:
        l = m1
    else:
        r = m2

best_threshold2 = l

# %%
acc3 = solveQ3(best_threshold2)
acc3

# %% [markdown]
# # --------------------------------- problem 4 -----------------------------------

# %%
def solveQ4(threshold1, threshold2):
    ans1 = set()
    for user, book, _ in newRatingsValid:
        B = usersPerItem[book] #all users who read the book
        maxSim = 0
        for bprime in itemsPerUser[user]:
            if bprime == book:
                continue
            B2 = usersPerItem[bprime]
            sim = Jaccard(B, B2)
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

#idea is u just use 2 threshold instead of 1 for the function and find the maximum 
    


# %%
best_threshold3 = scipy.optimize.fmin(lambda x: -solveQ4(x[0], x[1]), [0.699 ,0.6835848193872885 ])

# %%
best_threshold3

# %%
acc4 = solveQ4(best_threshold3[0], best_threshold3[1])

# %%
acc4

# %%
answers['Q3'] = acc3
answers['Q4'] = acc4

# %%
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

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

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
assert type(answers['Q5']) == str

# %%
##################################################
# Rating prediction                              #
##################################################

# %%
### Question 6

# %%
    
alpha = numpy.mean([r for _, _, r in ratingsTrain])
beta_user = defaultdict(float)
beta_book = defaultdict(float)
#referenced formula from 10/22 slides.  

for _ in range(50):
    alpha = sum(rating - (beta_user[user] + beta_book[book]) for user, book, rating in ratingsTrain) / len(ratingsTrain)
    for user, items in ratingsPerUser.items():
        beta_user[user] = sum(rating - (alpha + beta_book[book]) for book, rating in items) / (1 + len(items))
    for book, items in ratingsPerItem.items():
        beta_book[book] = sum(rating - (alpha + beta_user[user]) for user, rating in items) / (1 + len(items))

valid_error = [(rating - (alpha + beta_user.get(user, 0) + beta_book.get(book, 0))) ** 2 for user, book, rating in ratingsValid]
validMSE = numpy.mean(valid_error).item()

         

# %%
validMSE

# %%
answers['Q6'] = validMSE

# %%
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
maxUser = str(max(beta_user, key=beta_user.get))
maxBeta = beta_user[maxUser]
minUser = str(min(beta_user, key=beta_user.get))
minBeta = beta_user[minUser]


# %%
answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]
answers['Q7']

# %%
assert [type(x) for x in answers['Q7']] == [str, str, float, float]

# %%
### Question 8

# %%
def solve(lambda_reg):
    alpha = numpy.mean([r for _, _, r in ratingsTrain])
    beta_user = defaultdict(float)
    beta_book = defaultdict(float)
    #referenced formula from 10/22 slides.  

    for _ in range(10):
        alpha = sum(rating - (beta_user[user] + beta_book[book]) for user, book, rating in ratingsTrain) / len(ratingsTrain)
        for user, items in ratingsPerUser.items():
            beta_user[user] = sum(rating - (alpha + beta_book[book]) for book, rating in items) / (lambda_reg + len(items))
        for book, items in ratingsPerItem.items():
            beta_book[book] = sum(rating - (alpha + beta_user[user]) for user, rating in items) / (lambda_reg + len(items))

    valid_error = [(rating - (alpha + beta_user.get(user, 0) + beta_book.get(book, 0))) ** 2 for user, book, rating in ratingsValid]
    validMSE = numpy.mean(valid_error).item()
    return validMSE

# %%
l, r, eps = 0, 100, 1e-4

while(r-l > eps):
    m1 = l + (r-l)/3
    m2 = r - (r-l)/3
    print(m1, m2)
    if solve(m1) < solve(m2):
        r = m2
    else:
        l = m1

print(l, solve(l))

lamb = l
validMSE = solve(l)

# %%
answers['Q8'] = (lamb, validMSE)

# %%
assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])

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

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



