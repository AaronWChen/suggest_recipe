"""
This script is meant to be deployed on Amazon AWS and is designed to go to 
www.food.com/recipe, scroll through the endless scroll, and scrape the 
recipe-id and recipe-url attributes from the fd-tile.fd-recipe classes.

It will store these values into a database file on AWS.
"""

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
from typing import List
import json
from sqlalchemy import create_engine
import psycopg2
import logging
# from pymongo import MongoClient

# Prepare log handling
logger = logging.getLogger('food_scraper')
hdlr = logging.FileHandler('logs/food_scraper.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# connect to psql db
engine = create_engine("postgresql:///recipe_list")

# load selenium and Chrome options
app_path = "/usr/bin/chromedriver"
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(executable_path=app_path, chrome_options=chrome_options)

# URL of all recipes
url = "https://www.food.com/recipe"

# send Chrome to the URL
driver.get(url)
time.sleep(15)

# class name of interest
desired_class_name = 'fd-tile.fd-recipe'

# find and click button to begin load of endless scrolling
py_button = driver.find_element_by_class_name("gk-aa-load-more")
py_button.click()
time.sleep(10)
SCROLL_PAUSE_TIME = 5

print("Begin scroll and scrape")
iter = 0
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    iter += 1

    # Every 10 iterations, or 100 recipes, store the recipe-ids and recipe-urls
    # in a database on AWS RDS
    if iter % 10 == 0:
        hrefs = driver.find_elements_by_class_name(desired_class_name)

        list_of_ids = [href.get_attribute('data-id') for href in hrefs]
        list_of_links = [href.get_attribute('data-url') for href in hrefs]

        links_df = pd.DataFrame({'recipe-id': list_of_ids,
                                'recipe-url': list_of_links})

        links_df.to_sql(name='recipes',
                        con=engine,
                        if_exists='replace',)

    if new_height == last_height:
        break
    last_height = new_height

hrefs = driver.find_elements_by_class_name(desired_class_name)

list_of_ids = [href.get_attribute('data-id') for href in hrefs]
list_of_links = [href.get_attribute('data-url') for href in hrefs]

links_df = pd.DataFrame({'recipe-id': list_of_ids,
                        'recipe-url': list_of_links})

links_df.to_sql(name='recipes',
                con=engine,
                if_exists='replace',)
