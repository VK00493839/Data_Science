# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:07:03 2020

@author: VINAY KUMAR REDDY
"""

# SMSSpamCollection dataset
# cleaning the data by removing all punctuations and lowering the alphabets in the
# given message field, stemming the words and applying BOW
# one_hot encoding on label feature by applying get_dummies
# Applying Naive Bayes classifier to predict whether the message is ham/spam
# NB classifier works better for text data classification


import pandas as pd
import nltk, re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

ps = PorterStemmer()
# lemm = WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', " ", messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))
    

# creating BOW Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'], drop_first=True) # to get rid of dummy variable trap

# Train Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Training the model using MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

# Model Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_pred, y_test))
print(accuracy_score(y_pred, y_test))