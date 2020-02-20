# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:39:51 2020

@author: Szafran
"""

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the diabetes dataset
columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split() #declaring the columns names
diabetes = datasets.load_diabetes() #call the diabetes dataset from sklearn
df = pd.DataFrame(diabetes.data, columns=columns) #load the dataset as a pd.DataFrame with specified columns names
y = diabetes.target #define the target variable (dependent variable) as y

#create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

#Let's plot the model:

#The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

#And let's print the accuracy score:
print(f"Score {model.score(X_test, y_test)}")

# train_test_split has its flaws and it's described in onenote.