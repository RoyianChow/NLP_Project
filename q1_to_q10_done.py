#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:10 2023
@author: Theresa Bennett
@author: Melody Wan Wan
@author: Aaron Kai Chung Chan
@author: Vicky Mak Wing Ki
@author: Royian Chowdhury
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import numpy as np
import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#A)1
# Set up file path and names
vectorizer = CountVectorizer()
filename = 'Youtube02-KatyPerry.csv'
path = '/Users/royian_chowdhury/Desktop/Intro_to_AI/NLP_Project/'
fullpath = os.path.join(path, filename)

# Load data into DataFrame
comment_df = pd.read_csv(fullpath)
comment_df = comment_df.drop(['COMMENT_ID', 'DATE','AUTHOR'], axis=1)
# question 6
comment_df = comment_df.sample(frac=1)
# Carry out some basic data exploration and present your results.
#print("\nInformation of data:")
#print(comment_df.info())

#print("First 5 records:")
#print(comment_df.head(5))

#print("First 5 comments")
#print(comment_df["CONTENT"].head(5))

# Convert all text to lowercase
comment_df['CONTENT'] = comment_df['CONTENT'].apply(lambda x: x.lower())
# Tokenize the text
comment_df['CONTENT'] = comment_df['CONTENT'].apply(word_tokenize)
    
# Remove stop words
stop_words = set(stopwords.words('english'))

comment_df['CONTENT'] = comment_df['CONTENT'].apply(lambda x: [word for word in x if word not in stop_words])


# Lemmatize the text
lemmatizer = WordNetLemmatizer()
lemmas = []

def get_pos(nltk_pos):
    if nltk_pos.startswith('J'): # Adjectives
        return "a"
    elif nltk_pos.startswith('V'): # Verb
        return "v"
    elif nltk_pos.startswith('N'): # Noun
        return "n"
    elif nltk_pos.startswith('R'): # Adverb
        return "r"
    else:
        return "n"  # default to noun if the POS tag is not recognized
    
for comment in comment_df["CONTENT"]:
    tagged_input = pos_tag(comment)
    temp_lemmas = []
    for word, tag in tagged_input:
        inputpos = get_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=inputpos)
        temp_lemmas.append(lemma)
    lemmas.append(' '.join(temp_lemmas))

#print(lemmas)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)
# Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding

cols = vectorizer.get_feature_names_out()
#print(X)
# Convert the text to a bag-of-words representation
bow = pd.DataFrame(X.toarray(),columns=cols)
print(bow)


#
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(bow)
print(train_tfidf)

# split into 75% training and 25% testing sets. Since the data is shuffled in the beginning, we only need to select the first 75% of data.
train_size = int(len(comment_df) * 0.75)
train_data = comment_df[:train_size]
test_data = comment_df[train_size:]

X_train = train_tfidf[:train_size]
X_test = train_tfidf[train_size:]
y_train = train_data['CLASS']
y_test = test_data['CLASS']

#print("X_Train")
#print(X_train)

#Train model
model = MultinomialNB().fit(X_train, y_train)

# Cross validate the model on the training data using 5-fold
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5)
# Print the mean results of model accuracy
print(f"Accuracy: {scores.mean()}")

# Test the model on the test data
y_pred = model.predict(X_test)
# Print the confusion matrix and the accuracy of the model
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")
# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#print(train_tfidf)
# There are a lot of "words" that appear only once. These words can be 1. webpage link; and 2. words with repeating letters such as "pleaaaase"

#comment_names = []
#for i in range(X.shape[0]):
#   comment_names.append('Comment-' + str(i+1))

#print("\nTerm matrix:")
#formatted_text = '{:>12}' * (len(comment_names) + 1)
#print('\n', formatted_text.format("Word", *comment_names), '\n')

#for i, word in enumerate(cols):
#    row = X.T.getrow(i)
#    output = [word] + [str(freq) for freq in row.toarray()[0]]
#     print(formatted_text.format(*output))
    
# printing out the term matrix is not too useful as it is too big, but it can be looked into details when necessary
