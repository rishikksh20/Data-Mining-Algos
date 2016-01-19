# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:43:17 2016

@author: rishikesh
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import  linear_model
import csv

firstLine=True
firstfeatureName=''
secondfeatureName=''
thirdfeatureName=''
fourthfeatureName=''
MagazineName=[]
Adrevenue=[]
AdPages=[]
SubRevenue=[]
NewsRevenue=[]
MagzineData_X=[]
MagzineData_Y=[]
# Load the MagzineData dataset

with open('/home/rishikesh/Dev/Python/Data SCientist/Machine learning algos/DataSets/magazines.csv','r') as csvfile:
    MagzineData=csv.reader(csvfile, delimiter=',')
    for row in MagzineData:
        if firstLine :
            firstfeatureName=row[1]
            secondfeatureName=row[2]
            thirdfeatureName=row[3]
            fourthfeatureName=row[4]
            firstLine=False
            continue
        MagazineName.append(row[0])
        MagzineData_X.append(float(row[1]))
        MagzineData_Y.append(float(row[2]))



# Split the data into training/testing sets
MagzineData_X_train = MagzineData_X[:-102]
MagzineData_X_test = MagzineData_X[-102:]

# Split the targets into training/testing sets
MagzineData_y_train = MagzineData_Y[:-102]
MagzineData_y_test = MagzineData_Y[-102:]

# Create linear regression object
regr = linear_model.LinearRegression()
MagzineData_X_train=np.reshape(MagzineData_X_train,(102,1))
MagzineData_y_train=np.reshape(MagzineData_y_train,(102,1))
MagzineData_X_test=np.reshape(MagzineData_X_test,(102,1))
MagzineData_y_test=np.reshape(MagzineData_y_test,(102,1))
# Train the model using the training sets
regr.fit(MagzineData_X_train, MagzineData_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(MagzineData_X_test) - MagzineData_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(MagzineData_X_test, MagzineData_y_test))

# Plot outputs
plt.scatter(MagzineData_X_test, MagzineData_y_test,  color='black')
plt.plot(MagzineData_X_test, regr.predict(MagzineData_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()