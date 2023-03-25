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


import pandas as pd
import os


#A)1
# Set up file path and names
filename = 'Youtube02-KatyPerry.csv'
path = '/Users/royian_chowdhury/Desktop/Intro_to_AI/NLP_Project'
fullpath = os.path.join(path, filename)

# Load data into DataFrame
comment_df = pd.read_csv(fullpath)

print(comment_df)