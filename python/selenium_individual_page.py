from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import time
import requests
import pandas as pd
from typing import List
import json
from pymongo import MongoClient
from datetime import datetime

url = "https://www.food.com/recipe"
app_path = "/usr/local/bin/chromedriver"
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(executable_path = app_path, chrome_options=chrome_options)
driver.get(url)

# class SuperChrome(Chrome):
#     """References find_elements)by_css_selector in fewer words"""
#     def select(self, *args, **kwargs):
#         return self.find_elements_by_css_selector(*args, **kwargs)
#     def select_one(self, *args, **kwargs):
#         return self.find_element_by_css_selector(*args, **kwargs)

time.sleep(2)

# "div.gk-tile-basic h3 a"
# This selects 'a' tags, but you must get the href attributes off of them to get
# the links

query_string = """div[class=container-sm-md]""" 
# gk-tile-content]""" #:not([class*='featured-tiles']) 
# tile-stream 
# fd-tile.fd-recip"""

# """div.container-sm-md.gk-tile-content 
# div.tile-stream 
# div.fd-tile.fd-recipe"""

query_string_test = 'fd-tile.fd-recipe'

py_button = driver.find_element_by_class_name("gk-aa-load-more")
py_button.click()

time.sleep(15)

SCROLL_PAUSE_TIME = 3
print("Begin scroll and scrape")
iter = 0
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Extract the recipe-ids and recipe-urls currently obtainable
    hrefs = driver.find_elements_by_class_name(query_string_test)

    list_of_ids = [href.get_attribute('data-id') for href in hrefs]
    list_of_links = [href.get_attribute('data-url') for href in hrefs]

    links_df = pd.DataFrame({'recipe-id': list_of_ids,
                            'recipe-url': list_of_links})

    links_df.to_csv(f"write_data/recipes_page_{iter}.csv")

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    iter += 1
    print(f"Scrape counter: {iter}")

    if new_height == last_height:
        break
    last_height = new_height

hrefs = driver.find_elements_by_class_name(query_string_test)

list_of_ids = [href.get_attribute('data-id') for href in hrefs]
list_of_links = [href.get_attribute('data-url') for href in hrefs]

links_df = pd.DataFrame({'recipe-id': list_of_ids,
                        'recipe-url': list_of_links})

links_df.to_csv("write_data/all_recipes.csv")