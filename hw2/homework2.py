# %%
import numpy as np
import urllib
import scipy.optimize
import random
from sklearn import linear_model
import gzip
from collections import defaultdict

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(books, N):
    assert len(books) == N
    assert [type(float(x)) for x in books] == [float]*N

# %%
f = open("5year.arff", 'r')

# %%
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
answers = {} # Your answers

# %%
def accuracy(predictions, y):
    return sum([0 if predictions != y else 1 for predictions, y in zip(predictions, y)]) / len(y)

# %%
from sklearn.metrics import balanced_accuracy_score
def BER(predictions, y):
    return 1 - balanced_accuracy_score(y, predictions).item()

# %%
### Question 1

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)
matrix = confusion_matrix(y, mod.predict(X))
_ = ConfusionMatrixDisplay.from_estimator(mod , X, y)

pred = mod.predict(X)
acc1 = accuracy(pred, y)
ber1 = BER(pred, y)
print(acc1, ber1)

# %%
answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

matrix = confusion_matrix(y, mod.predict(X))
_ = ConfusionMatrixDisplay.from_estimator(mod , X, y)

pred = mod.predict(X)
acc2 = accuracy(pred, y)
ber2 = BER(pred, y)
print(acc2, ber2)

# %%
answers['Q2'] = [acc2, ber2]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
random.seed(3)
random.shuffle(dataset)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
len(Xtrain), len(Xvalid), len(Xtest)

# %%
model = linear_model.LogisticRegression(C=1, class_weight='balanced')
model.fit(Xtrain, ytrain)
matrix = confusion_matrix(ytrain, model.predict(Xtrain))
_ = ConfusionMatrixDisplay.from_estimator(model , Xtrain, ytrain)

matrix_valid = confusion_matrix(yvalid, model.predict(Xvalid))
_ = ConfusionMatrixDisplay.from_estimator(model , Xvalid, yvalid)

matrix_test = confusion_matrix(ytest, model.predict(Xtest))
_ = ConfusionMatrixDisplay.from_estimator(model , Xtest, ytest)



# %%
berTrain = BER(model.predict(Xtrain), ytrain)
berValid = BER(model.predict(Xvalid), yvalid)
berTest = BER(model.predict(Xtest), ytest)

# %%
answers['Q3'] = [berTrain, berValid, berTest]

# %%
assertFloatList(answers['Q3'], 3)

# %%
### Question 4

# %%
models = []
for CC in [10**k for k in range(-4, 5)]:
    model = linear_model.LogisticRegression(C=CC, class_weight='balanced')
    model.fit(Xtrain, ytrain)
    models.append(model)
berList = [BER(model.predict(Xvalid), yvalid) for model in models]

# %%
answers['Q4'] = berList
print(answers['Q4'])

# %%
assertFloatList(answers['Q4'], 9)

# %%
### Question 5

# %%
#ber lower = better
bestC = 10**((-4)+np.argmin(berList).item())
ber5 = berList[np.argmin(berList).item()]
print(ber5)

# %%
answers['Q5'] = [bestC, ber5]

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
dataTrain = dataset[:9000]
dataTest = dataset[9000:]

# %%
dataTrain[:10]

# %%
#changed the names to be more intuitive for me
bookToUser = defaultdict(set) # Maps an book to the users who rated it
userToBook = defaultdict(set) # Maps a user to the books that they rated
userToReview = defaultdict(list)
bookToReview = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/book pair

for d in dataTrain:
    user, book, rating = d['user_id'], d['book_id'], d['rating']
    bookToUser[book].add(user)
    userToBook[user].add(book)
    userToReview[user].append(d)
    bookToReview[book].append(d)
    ratingDict[(user,book)] = rating
    

# %%
def Jaccard(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

# %%
def mostSimilar(i, N):
    users = bookToUser[i]
    similarities = [
        (Jaccard(users, bookToUser[other]), other) for other in bookToUser if other != i
    ]
    return sorted(similarities, reverse=True)[:N]
    

# %%


# %%
answers['Q6'] = mostSimilar('2767052', 10)
print(answers['Q6'])

# %%
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

# %%
### Question 7

# %%

bookAvg = {b: sum([d['rating'] for d in bookToReview[b]]) / len(bookToReview[b]) if len(bookToReview[b]) > 0 else 0 for b in bookToReview}
ratingmean = sum([d['rating'] for d in dataTest]) / len(dataTest)

# %%
from sklearn.metrics import root_mean_squared_error
#referenced from 4.3.5 from textbook 
#we are predicting using book for this one
def predictRating(user, book):
    ratings, sims = [], []
    for d in userToReview[user]:
        j = d['book_id']
        if j==book: continue
        ratings.append(d['rating'] - bookAvg[j])
        sims.append(Jaccard(bookToUser[j], bookToUser[book])) #predict w book
    if sum(sims) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, sims)]
        return sum(weightedRatings) + bookAvg[book] / sum(sims)
    else:
        return ratingmean

    
mse7 = root_mean_squared_error([d['rating'] for d in dataTest], [predictRating(d['user_id'], d['book_id']) for d in dataTest])**2
mse7 = mse7.item()

# %%
answers['Q7'] = mse7
mse7

# %%
assertFloat(answers['Q7'])

# %%
### Question 8

# %%
userAvg = {u: sum([d['rating'] for d in userToReview[u]]) / len(userToReview[u]) if len(userToReview[u]) > 0 else 0 for u in userToReview}
ratingmean = sum([d['rating'] for d in dataTest]) / len(dataTest)

# %%
#referenced from 4.3.5 from textbook 
#we are predicting using user for this one

def predictRating(user, book):
    ratings, sims = [], []
    for d in bookToReview[book]:
        j = d['user_id']
        if j==user: continue
        ratings.append(d['rating'] - userAvg[j])
        sims.append(Jaccard(userToBook[j], userToBook[user])) #predict w user
    if sum(sims) > 0:
        weightedRatings = [(x*y) for x,y in zip(ratings, sims)]
        return sum(weightedRatings) + userAvg[user] / sum(sims)
    else:
        return ratingmean
    
mse8 = sum([(predictRating(d['user_id'], d['book_id']) - d['rating'])**2 for d in dataTest]) / len(dataTest)

# %%
answers['Q8'] = mse8
mse8

# %%
assertFloat(answers['Q8'])

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



