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

app_path = "/usr/local/bin/chromedriver"
#pc_app_path = 
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = Chrome(executable_path = app_path, chrome_options=chrome_options)

for url in urls_df['recipe-url']:
    driver.get(url)

    time.sleep(2)

    recipe_title = driver.find_element_by_class_name("recipe-title")
    title_text = recipe_title.get_attribute('Title')

    print(title_text)