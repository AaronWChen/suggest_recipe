from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import time
import requests
import pandas as pd
from typing import List
import json
from pymongo import MongoClient
from datetime import datetime

all_urls_df = pd.read_csv("write_data/test_run.csv")

urls_df = all_urls_df.drop([0,1])
urls_df = urls_df[:5]

app_path = "/usr/local/bin/chromedriver"
#pc_app_path = 
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(executable_path = app_path, chrome_options=chrome_options)

for url in urls_df['recipe-url']:
    driver.get(url)

    time.sleep(2)

    # Scrape the title of the recipe
    recipe_title = driver.find_element_by_tag_name('h1').text


    # Scrape the creator-provided description
    descrip = driver.find_element_by_class_name('print-only recipe-layout__description').text


    # Scrape the recipe author
    author_name = driver.find_element_by_class_name('recipe-details__author-link theme-color').text
    author_url = driver.find_element_by_class_name('recipe-details__author')
      need href

    # Scrape the star rating from the style and the number of reviews from the
    # text
    star_rating = driver.find_element_by_class_name("stars-rate__filler")
    rating_val = star_rating.get_attribute("style")

    review_count = driver.find_element_by_class_name("reviews-count").text


    # Scrape the recipe facts (READY IN time, SERVING count, UNITS style,
    # ) from the page
    rec_facts = driver.find_element_by_class_name("recipe-facts").text

    serving_tag = "recipe-facts__details recipe-facts__servings"
    num_servings = driver.find_element_by_class_name(serving_tag)

    ingredients_list_query = "recipe-ingredients__ingredient"
    ingredients_list = driver.find_elements_by_class_name(ingredients_list_query).text


    directions_list_query = "recipe-directions__steps"
    directions_list = driver.find_elements_by_class_name(directions_list_query).text
