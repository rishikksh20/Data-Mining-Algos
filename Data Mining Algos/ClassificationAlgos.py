# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:30:14 2016

@author: rishikesh
"""

# Gaussian Naive Bayes
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()


#############################################
import pandas
data_df = pandas.read_csv('Data Mining Algos/train.csv')
targetData=data_df.target
Data=data_df.drop(['id','target'],axis=1)
row_len=Data.feat_1.size
half_len=row_len/2
train=data_df.loc[data_df.id%2==0]
target=train.target
train=train.drop(['id','target'],axis=1)
test=data_df.loc[data_df.id%2==1]
expected=test.target
test=test.drop(['id','target'],axis=1)


model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))

print(train.id.size)


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(train,target)

predicted = model.predict(test)
print(accuracy_score(predicted,expected))


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train,target)

predicted = model.predict(test)
print(accuracy_score(predicted,expected))


from sklearn.svm import SVC

model = SVC()
model.fit(train,target)

predicted = model.predict(test)
print(accuracy_score(predicted,expected))


from sklearn.linear_model import LogisticRegression
# load the iris datasets
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(train,target)

predicted = model.predict(test)
print(accuracy_score(predicted,expected))


