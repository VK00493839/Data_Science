# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:26:59 2020

@author: VINAY KUMAR REDDY
"""

#importinglibraries
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

# performing Hyperparameter optimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Now let's create ANN Model
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization
# from keras.layers import PReLU, ReLU, ELU
from keras.activations import relu, sigmoid
from keras.layers import Dropout


def create_model(layers, activation):
    model = Sequential()
    # creating input and hidden layers
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    # adding output layer
    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    # compiling our model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [[20],[40,20],[45,30,15]]
activations = ['sigmoid','relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128,256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_result = grid.fit(X_train, y_train)


# Model best results
print(grid_result.best_score_, grid_result.best_params_)

# Predicting the test set results
pred_y = grid.predict(X_test)
y_pred = (pred_y>0.5)


# confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Claculating the Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))














