# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 01:14:03 2016

@author: rishikesh
"""
# Read .csv file using pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  linear_model
import pandas as pd

firstLine=True
firstfeatureName=''
secondfeatureName=''
thirdfeatureName=''
fourthfeatureName=''
MagazineName=[]
#Adrevenue=[]
#AdPages=[]
#SubRevenue=[]
#NewsRevenue=[]


# Load the MagzineData dataset
data = pd.read_csv('/home/rishikesh/Dev/Python/Data SCientist/Machine learning algos/DataSets/magazines.csv')

MagzineData_X=data['AdRevenue'].astype(float)
MagzineData_Y=data['AdPages'].astype(float)
#with open('/home/rishikesh/Dev/Python/Data SCientist/Machine learning algos/DataSets/magazines.csv','r') as csvfile:
#    MagzineData=csv.reader(csvfile, delimiter=',')
#    for row in MagzineData:
#        if firstLine :
#            firstfeatureName=row[1]
#            secondfeatureName=row[2]
#            thirdfeatureName=row[3]
#            fourthfeatureName=row[4]
#            firstLine=False
#            continue
#        MagazineName.append(row[0])
#        MagzineData_X.append(float(row[1]))
#        MagzineData_Y.append(float(row[2]))



# Split the data into training/testing sets
MagzineData_X_train =MagzineData_X[:-102]
MagzineData_X_test = MagzineData_X[-102:]

# Split the targets into training/testing sets
MagzineData_y_train = MagzineData_Y[:-102]
MagzineData_y_test = MagzineData_Y[-102:]

# Create linear regression object
regr = linear_model.LinearRegression()


# Reshape datas to array
MagzineData_X_train=np.reshape(MagzineData_X_train,(102,1))
MagzineData_y_train=np.reshape(MagzineData_y_train,(102,1))
MagzineData_X_test=np.reshape(MagzineData_X_test,(102,1))
MagzineData_y_test=np.reshape(MagzineData_y_test,(102,1))
MagzineData_X=np.reshape(MagzineData_X,(204,1))
MagzineData_Y=np.reshape(MagzineData_Y,(204,1))
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
plt.scatter( MagzineData_Y,MagzineData_X,  color='black')
plt.plot(regr.predict(MagzineData_X), MagzineData_X, color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()