"""
This script explores the dataset provided by the Kaggle listing here: 
https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions

The values in the csv should be stored in a database for easier retrieval/
access.

MVP: This is similar to the work done here:
https://github.com/majumderb/recipe-personalization
but uses NLTK instead of PyTorch to do NLP.
"""

# Import necessary libraries and modules
import json
import csv
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import string

# Open the recipe csv 
raw_file_str = "../food-com-recipes-and-user-interactions/RAW_recipes.csv"
df_file = pd.read_csv(raw_file_str)

# Generate stop words, punctuation, and lemmatizer for NLP
stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
lemmatizer = WordNetLemmatizer()


# https://stackoverflow.com/questions/5286541/how-can-i-flatten-lists-without-splitting-strings
def _flatten(list_of_lists):
    for x in list_of_lists:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in _flatten(x):
                yield y
        else:
            yield x