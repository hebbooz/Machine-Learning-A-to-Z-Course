# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset = pd.read_csv('titanic.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Identify the categorical features in your dataset that need to be encoded. You can store these feature names in a list for easy access later
# print(dataset.dtypes)
cat_features = ['Sex', 'Embarked', 'Pclass']

# Implement an instance of the ColumnTransformer class
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_features)], remainder='passthrough')


# Apply the fit_transform method on the instance of ColumnTransformer
X = np.array(ct.fit_transform(dataset))


# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])

# Print the updated matrix of features and the dependent variable vector
print(X)
print(y)
