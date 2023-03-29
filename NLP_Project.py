#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:52:10 2023

@author: Theresa
@author: Melody
@author: Aaron
@author: Vicky
@author: Royian



"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
#A)1
# Set up file path and names
vectorizer = CountVectorizer()
filename = 'Youtube02-KatyPerry.csv'
path = '/Users/royian_chowdhury/Desktop/Intro_to_AI/NLP_Project'
fullpath = os.path.join(path, filename)

# Load data into DataFrame
comment_df = pd.read_csv(fullpath)

comment_df = comment_df.drop(['COMMENT_ID', 'DATE','AUTHOR'], axis=1)

# Convert all text to lowercase
comment_df['CONTENT'] = comment_df['CONTENT'].apply(lambda x: x.lower())
# Tokenize the text
comment_df['CONTENT'] = comment_df['CONTENT'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
comment_df['CONTENT'] = comment_df['CONTENT'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the text to a bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comment_df['CONTENT'].apply(lambda x: ' '.join(x)))

print(X)