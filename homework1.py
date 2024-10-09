# %%
import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy as np
import random
import gzip
import math

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
len(dataset)

# %%
answers = {} # Put your answers to each question in this dictionary

# %%
dataset[:5]

# %%
### Question 1

# %%
def feature(datum): #predict rating from number of exclamation points
    feat = [1, datum['review_text'].count('!')]
    return feat
    

# %%
X = [feature(d) for d in dataset] #how many ! in a review
Y = [d['rating'] for d in dataset] #overall rating

# %%
X[:10]


# %%
Y[:10]

# %%
model = linear_model.LinearRegression().fit(X, Y)
theta0, theta1 = model.intercept_.item(), model.coef_[1].item()

theta0, theta1

# %%
from sklearn.metrics import root_mean_squared_error
mse = root_mean_squared_error(Y, model.predict(X))**2
mse = mse.item()

# %%
theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None) #just trying out another way

# %%
theta

# %%
answers['Q1'] = [theta0, theta1, mse]

# %%
assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

# %%
### Question 2

# %%
def feature(datum):
    feat = [1, len(datum['review_text']), datum['review_text'].count('!')]
    return feat

# %%
X = [feature(d) for d in dataset] #length of review, number of ! in review
Y = [d['rating'] for d in dataset] #overall rating
X[:10]


# %%
model = linear_model.LinearRegression().fit(X, Y)
theta0, theta1, theta2 = model.intercept_.item(), model.coef_[1].item(), model.coef_[2].item()
mse = root_mean_squared_error(Y, model.predict(X))**2
mse = mse.item()

# %%
answers['Q2'] = [theta0, theta1, theta2, mse]
answers['Q2']

# %%
assertFloatList(answers['Q2'], 4)

# %%
### Question 3

# %%
def feature(datum, deg): #poly from 1 to 5
    feat = [1] + [datum['review_text'].count('!')**i for i in range(1, deg+1)]
    return feat

# %%
X_train = [[feature(d, i) for d in dataset] for i in range(1, 6)]
Y = [d['rating'] for d in dataset]
models = [linear_model.LinearRegression().fit(X_train[i], Y) for i in range(5)]
mses = [root_mean_squared_error(Y, models[i].predict(X_train[i]))**2 for i in range(5)]
mses = [mse.item() for mse in mses]


# %%
mses

# %%
answers['Q3'] = mses

# %%
assertFloatList(answers['Q3'], 5)# List of length 5

# %%
### Question 4

# %%
training_set, test_set = dataset[:5000], dataset[5000:]

X_train = [[feature(d, i) for d in training_set] for i in range(1, 6)]
X_test = [[feature(d, i) for d in test_set] for i in range(1, 6)]

Y_train = [d['rating'] for d in training_set]
Y_test = [d['rating'] for d in test_set]

models = [linear_model.LinearRegression().fit(X_train[i], Y_train) for i in range(5)]
mses = [root_mean_squared_error(Y_test, models[i].predict(X_test[i]))**2 for i in range(5)]
mses = [mse.item() for mse in mses]

mses

# %%
answers['Q4'] = mses

# %%
assertFloatList(answers['Q4'], 5)

# %%
### Question 5

# %%
from sklearn.metrics import mean_absolute_error
theta0 = np.median(Y_test).item()
mae = mean_absolute_error(Y_test, [theta0] * len(Y_test)).item()

theta0, mae


# %%
answers['Q5'] = mae

# %%
assertFloat(answers['Q5'])

# %%
### Question 6

# %%
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

# %%
dataset[:5]

# %%
def feature(datum): #predict rating from number of exclamation points
    feat = [1, datum['review/text'].count('!')]
    return feat

# %%
X = [feature(d) for d in dataset]
Y = [d['user/gender'] == "Female" for d in dataset] #1 for female 

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model = linear_model.LogisticRegression().fit(X, Y)
matrix = confusion_matrix(Y, model.predict(X))
_ = ConfusionMatrixDisplay.from_estimator(model , X, Y)

TN, FP, FN, TP = matrix[0][0].item(), matrix[0][1].item(), matrix[1][0].item(), matrix[1][1].item()
BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
TN, FP, FN, TP, BER

# %%
answers['Q6'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q6'], 5)

# %%
### Question 7

# %%
model = linear_model.LogisticRegression(class_weight="balanced").fit(X, Y)
matrix = confusion_matrix(Y, model.predict(X))
_ = ConfusionMatrixDisplay.from_estimator(model , X, Y)

TN, FP, FN, TP = matrix[0][0].item(), matrix[0][1].item(), matrix[1][0].item(), matrix[1][1].item()
BER = 0.5 * (FP / (FP + TN) + FN / (FN + TP))
TN, FP, FN, TP, BER

# %%
answers["Q7"] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q7'], 5)

# %%
### Question 8

# %%
from sklearn.metrics import precision_score

predictions = model.predict_proba(X)[:, 1]

K_values = [1, 10, 100, 1000, 10000]
precisionList = []

for K in K_values:
    top_k = np.argsort(predictions)[-K:]
    precisionList.append(precision_score(Y, [i in top_k for i in range(len(Y))]).item())


# %%
answers['Q8'] = precisionList

# %%
assertFloatList(answers['Q8'], 5) #List of five floats

# %%
answers

# %%
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()

# %%



