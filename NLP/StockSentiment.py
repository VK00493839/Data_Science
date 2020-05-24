# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:02:18 2020

@author: VINAY KUMAR REDDY
"""


# steps followed
# reading the data.csv file
# splitting the data on date criteria
# cleaning the data (removing punctuations)
# convert all alphabets to lower case as we are converting data to BOW vector format
# concatenate all features as a single text for each record and store in headlines list
# Apply BOW and RandomForestClassifier on trainDataset
# Apply the same on testDataset and calculate the metrics


import nltk
import pandas as pd

df = pd.read_csv('Data.csv', encoding='ISO-8859-1')

train = df[df['Date']<'20150101']
test = df[df['Date']>'20141231']

# preprocessing the data
data = df.iloc[:,2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# renaming the columns for readability
new_index = [str(i) for i in range(25)]
data.columns = new_index

# converting the all alphabets into lower
for i in new_index:
    data[i] = data[i].str.lower()
    
# print(' '.join(str(x) for x in data.iloc[1,0:25]))
headlines = []
for row in range(len(train.index)): # looping over the train data record wise
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


    
# Applying BOW model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

countVector = CountVectorizer(ngram_range=(2,2))
trainDataset = countVector.fit_transform(headlines)

# implementing RandomForestClassifier
rf_Class = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_Class.fit(trainDataset, train['Label'])

# predict the test data
testData = []
for row in range(len(test.index)): # looping over the test data record wise
    testData.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
    
testDataset = countVector.fit_transform(testData)
pred = rf_Class.predict(testDataset)

# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
matrix = confusion_matrix(test['Label'], pred)
score = accuracy_score(test['Label'], pred)
report = classification_report(test['Label'], pred)