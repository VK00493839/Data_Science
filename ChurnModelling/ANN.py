# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:22:00 2020

@author: VINAY KUMAR REDDY
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# imprtring datasets
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# create dummy variables for Gender and Geography4
geography = pd.get_dummies(X['Geography'], drop_first=True)
gender = pd.get_dummies(X['Gender'], drop_first=True)

# concatenting these two columns to our dataset and deleting the Geography, Gender columns from original dataset
X = pd.concat([X, geography, gender], axis=1) #column wise
X.drop(['Geography', 'Gender'], axis=1, inplace=True)

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Now let's create ANN Model
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import PReLU, ReLU, ELU, LeakyReLU
from keras.layers import Dropout


# initializing Ann model
classifier = Sequential()

# Adding the Input layer and First hidden layer
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))

# Adding the Second hidden layer
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, nb_epoch=10)

# list of all data in history
print(model_history.history.keys())

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Claculating the Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))














