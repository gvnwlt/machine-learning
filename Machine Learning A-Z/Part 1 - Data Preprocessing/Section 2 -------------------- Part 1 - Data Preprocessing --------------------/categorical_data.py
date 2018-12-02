# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# TAKING CARE OF MISSING DATA
from sklearn.preprocessing import Imputer 
# identify any missing values of NaN
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# impute the values using mean value for all rows for columns 1 to 3
imputer = imputer.fit(X(:, 1:3))
X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding the categorical data 
# encoding the independant variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X = LabelEncoder() 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# encoding the dependant variable 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 
