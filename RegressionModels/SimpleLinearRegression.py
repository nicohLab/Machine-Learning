"""
SIMPLE LINEAR REGRESION
-----------------------
@autor: Nicol Huaraca
We'll learn to compute linear regression model using scikit-learn library
"""

# Import packages
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import linear_model

# Import data
df = pd.read_csv("data1.csv")

# view top 5
df.head(5)

# Select some features of the data
sub_df = df[['GASEXP', 'POP', 'INCOME', 'PUC']]
sub_df.head(5)

# Graph histograms
sub_df.hist()
plt.show

# Scatterplot 
plt.scatter(sub_df.INCOME, sub_df.GASEXP, color='red')
plt.xlabel('Income')
plt.ylabel('Gasexp')
plt.show()

# Divide of dataset in training and test data
rul = np.random.rand(len(sub_df)) < 0.6
train = sub_df[rul]
test = sub_df[~rul]

# Simple regression model
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['INCOME']])
train_y = np.asanyarray(train[['GASEXP']])

reg.fit (train_x, train_y)

# The coefficients
print('Coefficients: ', reg.coef_)
print('Intercept: ', reg.intercept_)

#Plot
plt.scatter(train.INCOME, train.GASEXP, color='red')
plt.plot(train_x, reg.coef_[0][0]*train + reg.intercept_[0], '-g')
plt.xlabel('Income')
plt.ylabel('Gasexp')
plt.show()