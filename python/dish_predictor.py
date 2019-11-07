""" This Python file calls the Edamam API with user-desired dish name and user-
specified cuisine style, generates a JSON from Edamam with recipes to analyze, 
reduces the 10 provided recipes to one list of ingredients to vectorize, loads 
the database generated by prepare_database.py, finds the top five similar 
recipes based on cosine similarity distances for each list of ingredients.
"""

import requests
import json
import csv
#import pymongo
from datetime import datetime
import joblib
import re
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

# Define all functions

def import_stored_files():
  # Load in the stored Epicurious database, TFIDF Vectorizer object to transform,
  # the input, and the TFIDF word matrix from joblib and created by 
  # prepare_database.py
  with open("joblib/recipe_dataframe.joblib", "rb") as fo:
    prepped = joblib.load("joblib/recipe_dataframe.joblib")

  with open("joblib/recipe_tfidf.joblib", "rb") as fo:
    ingred_tfidf = joblib.load("joblib/recipe_tfidf.joblib")

  with open("joblib/recipe_word_matrix.joblib", "rb") as fo:
    ingred_word_matrix = joblib.load("joblib/recipe_word_matrix.joblib")

  return prepped, ingred_tfidf, ingred_word_matrix


def transform_tfidf(ingred_tfidf, recipe):
  # This function takes in a TFIDF Vectorizer object and a recipe, then 
  # creates/transforms the given recipe into a TFIDF form

  recipe = [' '.join(recipe['ingredients'][0])]
  response = ingred_tfidf.transform(recipe)
  transformed_recipe = pd.DataFrame(response.toarray(),
                                    columns=ingred_tfidf.get_feature_names())
  return transformed_recipe


def filter_out_cuisine(ingred_word_matrix, 
                        X_df, 
                        cuisine_name, 
                        tfidf):
  # This function takes in the ingredient word matrix (from joblib), a 
  # dataframe made from the database (from joblib), the user inputted cuisine 
  # name, and the ingredient TFIDF Vectorizer object (from joblib) and returns
  # a word sub matrix that removes all recipes with the same cuisine as the 
  # inputted recipe.

  combo = pd.concat([ingred_word_matrix, X_df['imputed_label']], axis=1)
  filtered_ingred_word_matrix = combo[combo['imputed_label'] != cuisine_name].drop('imputed_label', 
                                                                    axis=1)
  return filtered_ingred_word_matrix


def find_closest_recipes(filtered_ingred_word_matrix, 
                          recipe_tfidf, 
                          X_df):
  # This function takes in the filtered ingredient word matrix from function
  # filter_out_cuisine, the TFIDF recipe from function transform_tfidf, and 
  # a dataframe made from the database (from joblib) and returns a Pandas 
  # DataFrame with the top five most similar recipes and a Pandas Series 
  # containing the similarity amount
  search_vec = np.array(recipe_tfidf).reshape(1,-1)
  res_cos_sim = cosine_similarity(filtered_ingred_word_matrix, search_vec)
  top_five = np.argsort(res_cos_sim.flatten())[-5:][::-1]
  
  recipe_ids = [filtered_ingred_word_matrix.iloc[idx].name for idx in top_five]
  suggest_df = X_df.loc[recipe_ids]
  proximity = pd.DataFrame(data=res_cos_sim[top_five], 
                            columns=['cosine_similarity'], 
                            index=suggest_df.index)
  full_df = pd.concat([suggest_df, proximity], axis=1)
  expand_photo_df = pd.concat([full_df.drop(["photo_data"], axis=1), 
                                full_df["photo_data"].apply(pd.Series)], axis=1)
  reduced = expand_photo_df[['title', 'url', 'filename', 'imputed_label', 'ingredients', 'cosine_similarity']].dropna(axis=1)
  return reduced

def find_similar_dishes(dish_name, cuisine_name):
  prepped, ingred_tfidf, ingred_word_matrix = import_stored_files()
  # This function calls the Edamam API, stores the results as a JSON, and 
  # stores the timestamp, dish name, and cuisine name/classification in a 
  # separate csv.
  now = datetime.now()
  dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

  api_base = "https://api.edamam.com/search?"

  # Level up:
  # Implement lemmatization using trained dataset on input in order to make 
  # future database be less likely to have redundant entries 
  # (e.g., taco vs tacos)

  q = f"q={dish_name}"

  # Level up:
  # Check a database of dishes to see if this query has been asked for already
  # If not, do an API call

  # Currently, just does an API call, may hit API limit if continuing with this
  # version
  f = open("../secrets/edamam.json","r")
  cred = json.load(f)
  f.close()

  app_id = cred["id"]
  app_id_s = f"&app_id=${app_id}"

  app_key = cred["key"]
  app_key_s = f"&app_key=${app_key}"
  
  # Level up: 
  # Explicitly ask for a few recipes using limiter and make an "average version"
  # of the input in order to get better results from the API call
  # limiter = "&from=0&to=4"
  # API currently defaults to returning 10

  api_call = api_base + q+ app_id_s + app_key_s #+ limiter

  resp = requests.get(api_call)

  if resp.status_code == 200:
    response_dict = resp.json()
    resp_dict_hits = response_dict['hits']
    
    # Store the API result into a JSON and the cuisine type and dish name into a 
    # csv
    # Heroku does not save files to directory
    # Can work with EC2
    with open(f"../write_data/{dt_string}_{dish_name}_edamam_api_return.json", "w") as f:
      json.dump(resp_dict_hits, f)

    fields = [dt_string, dish_name, cuisine_name]
    with open("../write_data/user_requests.csv", "a", newline='') as f:
      writer = csv.writer(f)
      writer.writerow(fields)

    urls = []
    labels = []
    sources = []
    ingreds = []

    for recipe in resp_dict_hits:
        recipe_path = recipe['recipe']
        urls.append(recipe_path['url'])
        labels.append(recipe_path['label'])
        sources.append(recipe_path['source'])
        ingreds.append([item['food'] for item in recipe_path['ingredients']])
        
    all_recipes = {'url': urls,
                  'label': labels, 
                  'source': sources, 
                  'ingredients': ingreds
                  }

    recipe_df = pd.DataFrame(all_recipes)

    one_recipe = []

    for listing in recipe_df['ingredients']:
        for ingred in listing:
            one_recipe.append(ingred.lower())
        
    one_recipe = list(set(one_recipe))

    query_df = pd.DataFrame(data={'name': dish_name, 'ingredients': [one_recipe], 'cuisine': cuisine_name})

    query_tfidf = transform_tfidf(ingred_tfidf=ingred_tfidf, recipe=query_df)
    query_matrix = filter_out_cuisine(ingred_word_matrix=ingred_word_matrix, 
                                      X_df=prepped, 
                                      cuisine_name=cuisine_name, 
                                      tfidf=ingred_tfidf)
                                      
    query_similar = find_closest_recipes(filtered_ingred_word_matrix=query_matrix, 
                                                                recipe_tfidf=query_tfidf, 
                                                                X_df=prepped)
    return query_similar.to_html()
    # reduced_query = query_similar[['title', 
    #                               'url', 
    #                               'filename',
    #                               'imputed_label',
    #                               'ingredients', 
    #                               'cosine_similarity']]
    # return reduced_query.to_html() #("../write_data/results.html")
    
  else:
    return("Error, unable to retrieve. Server response code is: ", 
          resp.status_code)