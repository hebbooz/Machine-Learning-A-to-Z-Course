# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
dataset = pd.read_csv("iris.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Separate features and target
from sklearn.model_selection import train_test_split

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    
# Print the scaled training and test sets
print("X_train:")
print(X_train)

print("X_test:")
print(X_test)

print("y_train")
print(y_train)

print("y_test")
print(y_test)