# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:31:49 2016

@author: rishikesh
"""
# Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics

# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
# Print the model after fit the training data
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
