# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:30:14 2016

@author: rishikesh
"""


import pandas
from sklearn.metrics import accuracy_score

# Pre-processing the Data
data_df = pandas.read_csv('train.csv')
train=data_df.loc[data_df.id%2==0]
target=train.target
train=train.drop(['id','target'],axis=1)
test=data_df.loc[data_df.id%2==1]
expected=test.target
test=test.drop(['id','target'],axis=1)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))

#KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))

# SVM
from sklearn.svm import SVC
model = SVC()
model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train,target)
predicted = model.predict(test)
print(accuracy_score(predicted,expected))


