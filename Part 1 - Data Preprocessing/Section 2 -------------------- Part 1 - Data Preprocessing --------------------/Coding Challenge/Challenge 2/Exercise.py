# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()

# Print the number of missing entries in each column
print(missing_data)

# Configure an instance of the SimpleImputer class
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(X[:, :]) 

# Apply the transform to the DataFrame
X[:, :] = imputer.transform(X[:, :])

#Print your updated matrix of features