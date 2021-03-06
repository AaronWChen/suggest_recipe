""" This python script takes the JSON and text file generated by input from an 
Edamam API search for a dish and the cuisine type, opens the JSON and text 
files, and creates a Pandas dataframe that can be transformed via TFIDF for
cosine similarity analysis."""


# Import necessary libraries
import json
import csv
import re
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from operator import itemgetter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

#input from search_q, cuisine_q from api_req.py

json_filename = "../write_data/test_api_hits.json"
with open(json_filename, 'r') as f:
    datastore = json.load(f)

urls = []
labels = []
sources = []
ingreds = []

for recipe in datastore:
    recipe_path = recipe['recipe']
    urls.append(recipe_path['url'])
    labels.append(recipe_path['label'])
    sources.append(recipe_path['source'])
    ingreds.append([item['food'] for item in recipe_path['ingredients']])
    
all_recipes = {'url': urls,
               'label': labels, 
               'source': sources, 
               'ingredients': ingreds,
               'cuisine': 'Mexican'
              }

recipe_df = pd.DataFrame(all_recipes)

recipe_df.to_json("../write_data/recipe_df.json")
