# simple linear regression 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# takes columns from 0 to end less than one (exclude last column, so 0 column)
# [number of years] 
X = dataset.iloc[:, :-1].values
# select column 1 (second column): 
# [salaries] 
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# y_test = ground_truth(actual)

# Feature Scaling: for simple linear regression the library takes care of this
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set results 
y_pred = regressor.predict(X_test)


# visualizing the training set results 
# show x and y training data plots given during model training (real values)
plt.scatter(X_train, y_train, color='red')
# show results of prediction during model training (results each epoch; prediction values)
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the training set results 
plt.scatter(X_test, y_test, color='red')
# same regression line will be generated so no need to change 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()