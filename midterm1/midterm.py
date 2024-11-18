# %%
import json
import gzip
import math
import numpy
from collections import defaultdict
from sklearn import linear_model
import random
import statistics

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
# From https://cseweb.ucsd.edu/classes/fa24/cse258-b/files/steam.json.gz
z = gzip.open("steam.json.gz")

# %%
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

# %%
z.close()

# %%
### Question 1

# %%
dataset[:10]

# %%
from sklearn.metrics import root_mean_squared_error
def MSE(y, ypred):
    return (root_mean_squared_error(y, ypred)**2).item()

# %%
def feat1(d):
    feat = [1, len(d['text'])] 
    return feat  

# %%
X = [feat1(d) for d in dataset]
y = [d['hours'] for d in dataset]

# %%
model = linear_model.LinearRegression()
model.fit(X, y)

# %%
mse1 = MSE(y, model.predict(X))

# %%
theta1 = model.coef_[1].item()

# %%
answers['Q1'] = [theta1, mse1] # Remember to cast things to float rather than (e.g.) np.float64

# %%
assertFloatList(answers['Q1'], 2)

# %%
answers['Q1']

# %%
### Question 2

# %%
dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]

# %%
model2 = linear_model.LinearRegression()
X = [feat1(d) for d in dataTrain]
y = [d['hours'] for d in dataTrain]
model2.fit(X, y)

mse2 = MSE(y, model2.predict(X))

mse2

# %%
under = 0
over = 0

Xtest = [feat1(d) for d in dataTest]
for d in dataTest:
    if d['hours'] < model2.predict([feat1(d)]):
        over += 1
    else:
        under += 1


# %%
print(under, over)

# %%
answers['Q2'] = [mse2, under, over]

# %%
assertFloatList(answers['Q2'], 3)

# %%
### Question 3

# %%
y2 = y[:]
y2.sort()
perc90 = y2[int(len(y2)*0.9)] # 90th percentile
X3a = []
y3a = []
for d in dataTrain:
    if d['hours'] <= perc90:
        X3a.append(feat1(d))
        y3a.append(d['hours'])


mod3a = linear_model.LinearRegression(fit_intercept=False)
mod3a.fit(X3a,y3a)
pred3a = mod3a.predict(Xtest)

# %%
under3a = 0
over3a = 0

for d in dataTest:
    if d['hours'] < mod3a.predict([feat1(d)]):
        over3a += 1
    else:
        under3a += 1

# %%
print(under3a, over3a)

# %%
# etc. for 3b and 3c

# %%
X3b = []
y3b = []
for d in dataTrain:
    X3b.append(feat1(d))
    y3b.append(d['hours_transformed'])


mod3b = linear_model.LinearRegression(fit_intercept=False)
mod3b.fit(X3b,y3b)
pred3b = mod3b.predict(Xtest)

# %%
under3b = 0
over3b = 0

for d in dataTest:
    if d['hours_transformed'] < mod3b.predict([feat1(d)]):
        over3b += 1
    else:
        under3b += 1

# %%
print(under3b, over3b)

# %%
median_length = numpy.median([len(d['text']) for d in dataTrain])
median_hours = numpy.median([d['hours'] for d in dataTrain])
print(median_length, median_hours)

# %%
theta0, theta1 = model2.intercept_, model2.coef_[1]

# %%
print(theta0, theta1)
newtheta1 = (median_hours-theta0)/median_length

# %%
print(newtheta1)

# %%
under3c, over3c = 0, 0

for d in dataTest:
    if d['hours'] < theta0 + newtheta1 * len(d['text']):
        over3c += 1
    else:
        under3c += 1


# %%
print(under3c, over3c)

# %%
answers['Q3'] = [under3a, over3a, under3b, over3b, under3c, over3c]

# %%
assertFloatList(answers['Q3'], 6)

# %%
### Question 4

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
Xtrain = [feat1(d) for d in dataTrain]
ytrain = [d['hours'] > median_hours for d in dataTrain]
Xtest = [feat1(d) for d in dataTest]
ytest = [d['hours'] > median_hours for d in dataTest]
mod = linear_model.LogisticRegression(C=1)
mod.fit(Xtrain,ytrain)
matrix = confusion_matrix(ytest, mod.predict(Xtest))
_ = ConfusionMatrixDisplay.from_estimator(mod, Xtest, ytest)

TN, FP, FN, TP = matrix.ravel()
BER = 0.5 * (FP/(FP+TN) + FN/(FN+TP))
print(TN, FP, FN, TP, BER)



# %%
answers['Q4'] = [TP.item(), TN.item(), FP.item(), FN.item(), BER.item()]

# %%
assertFloatList(answers['Q4'], 5)

# %%
### Question 5

# %%
answers['Q5'] = [FP.item(), FN.item()]

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
X2014 = []
y2014 = []
X2015plus = []
y2015plus = []
for d in dataTrain:
    if int(d['date'][:4]) <= 2014:
        X2014.append(feat1(d))
        y2014.append(d['hours'] > median_hours)
    else:
        X2015plus.append(feat1(d))
        y2015plus.append(d['hours'] > median_hours)

X2014test = []
y2014test = []
X2015plustest = []
y2015plustest = []

for d in dataTest:
    if int(d['date'][:4]) <= 2014:
        X2014test.append(feat1(d))
        y2014test.append(d['hours'] > median_hours)
    else:
        X2015plustest.append(feat1(d))
        y2015plustest.append(d['hours'] > median_hours)

# minset = min(len(X2014test), len(X2015plustest))


# %%
from sklearn.metrics import balanced_accuracy_score
def BER(predictions, y):
    return 1 - balanced_accuracy_score(y, predictions).item()

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(X2014,y2014)
BER_A = BER(mod.predict(X2014test), y2014test)
mod2 = linear_model.LogisticRegression(C=1)
mod2.fit(X2015plus,y2015plus)
BER_B = BER(mod2.predict(X2015plustest), y2015plustest)
BER_C = BER(mod.predict(X2015plustest), y2015plustest)
BER_D = BER(mod2.predict(X2014test), y2014test)


# %%
print(BER_A, BER_B, BER_C, BER_D)

# %%
answers['Q6'] = [BER_A, BER_B, BER_C, BER_D]

# %%
assertFloatList(answers['Q6'], 4)

# %%
### Question 7

# %%
def Jaccard(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2)) 

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataTrain:
    user, item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

# %%
jaccards = []
userset = set()
for d in dataTrain:
    userset.add(d['userID'])
firstPerson = itemsPerUser[dataTrain[0]['userID']]
for user in userset:
    if(user != dataTrain[0]['userID']):
        jaccards.append([Jaccard(itemsPerUser[user], firstPerson), user])

jaccards = sorted(jaccards, key=lambda x: x[0], reverse=True)
print(jaccards[:10])


# %%
answers['Q7'] = [jaccards[0][0], jaccards[9][0]]

# %%
assertFloatList(answers['Q7'], 2)

# %%
### Question 8

# %%
hours_transformed = defaultdict()
for d in dataTrain:
    hours_transformed[d['userID'], d['gameID']] = d['hours_transformed']
hours_transformed_median = numpy.median([d['hours_transformed'] for d in dataTrain])

# %%
hours_transformed

# %%
def predict_hours_transformed_by_user(user, item):
    top = 0
    bottom = 0
    users = usersPerItem[item]
    for v in users:
        if v == user: continue
        j = Jaccard(itemsPerUser[user], itemsPerUser[v])
        top += hours_transformed[v, item] * j
        bottom += j
    if not bottom:
        return hours_transformed_median
    return top / bottom


# %%
def predict_hours_transformed_by_item(user, item):
    top = 0
    bottom = 0
    items = itemsPerUser[user]
    for v in items:
        if v == item: continue
        j = Jaccard(usersPerItem[item], usersPerItem[v])
        top += hours_transformed[user, v] * j
        bottom += j
    if not bottom:
        return hours_transformed_median
    return top / bottom

# %%
MSEU = 0
for d in dataTest:
    user, item = d['userID'], d['gameID']
    pred = predict_hours_transformed_by_user(user, item)
    MSEU += (pred - d['hours_transformed'])**2
MSEU /= len(dataTest)

# %%
MSEI = 0
for d in dataTest:
    user, item = d['userID'], d['gameID']
    pred = predict_hours_transformed_by_item(user, item)
    MSEI += (pred - d['hours_transformed'])**2
MSEI /= len(dataTest)

# %%
print(MSEU, MSEI)

# %%
answers['Q8'] = [MSEU.item(), MSEI.item()]

# %%
assertFloatList(answers['Q8'], 2)

# %%
### Question 9

# %%
review_year = defaultdict()
for d in dataset:
    review_year[d['userID'], d['gameID']] = int(d['date'][:4])

# %%
def predict_hours_transformed_by_user2(user, item):
    top = 0
    bottom = 0
    users = usersPerItem[item]
    for v in users:
        if v == user: continue
        j = Jaccard(itemsPerUser[user], itemsPerUser[v])
        weight = math.exp(-abs(review_year[user, item] - review_year[v, item]))
        top += hours_transformed[v, item] * j * weight
        bottom += j * weight
    if not bottom:
        return hours_transformed_median
    return top / bottom

# %%
MSE9 = 0
for d in dataTest:
    user, item = d['userID'], d['gameID']
    pred = predict_hours_transformed_by_user2(user, item)
    MSE9 += (pred - d['hours_transformed'])**2
MSE9 /= len(dataTest)

# %%
answers['Q9'] = MSE9.item()

# %%
print(answers)

# %%
assertFloat(answers['Q9'])

# %%
if "float" in str(answers) or "int" in str(answers):
    print("it seems that some of your answers are not native python ints/floats;")
    print("the autograder will not be able to read your solution unless you convert them to ints/floats")

# %%
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



