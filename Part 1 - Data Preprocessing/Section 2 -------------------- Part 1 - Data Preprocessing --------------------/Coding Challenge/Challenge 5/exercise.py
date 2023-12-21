import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Wine Quality Red dataset
dataset = pd.read_csv("winequality-red.csv", delimiter=';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Separate features and target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print(X_train)
print("--------------------------------------------------")
print(X_test)