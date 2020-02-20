# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:57:55 2020

@author: Szafran
"""

from sklearn.model_selection import KFold
import numpy as np

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) #create an array
y = np.array([1, 2, 3, 4]) #create another array
kf = KFold(n_splits=3) #define the split - into 2 folds
kf.get_n_splits(X) #returns the number of splitting iterations in the cross-validation

print(kf)

#And let’s see the result — the folds:

for train_index, test_index in kf.split(X):
    print("Train: ", train_index, "Test: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
#As you can see, the function split the original data into different subsets of the data.
#Again, very simple example but I think it explains the concept pretty well.
#oczywiscie number of splits musi byc mniejsze niz ilosc obserwacji
  """
poniżej model z pliku dangers
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
  
    """
dobra od teraz sie bawimy
"""



from matplotlib import pyplot as plt    
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

#perform 6-fold cross validation
scores = cross_val_score(model, df, y, cv=6)
print("Cross-validated scores:", scores)

#As you can see, the last fold improved the score of the original model — from 0.485 to 0.569. Not an amazing result, but hey, we’ll take what we can get :)

#make a cross validated predictions
predictions = cross_val_predict(model, df, y, cv=6) #there's probably a mistka as the cross validation should only be performed on the training set (which is splitted into training and validation), see a screen in onenote
plt.scatter(y, predictions) #6 razy więcej punktów niż za 1 razem.

#sprawdźmy R^2 modelu
accuracy = metrics.r2_score(y, predictions)
print("Cross-predicted accuracy:", accuracy)