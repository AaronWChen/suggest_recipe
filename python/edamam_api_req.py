""" This Python file calls the Edamam API to return a JSON file with desired
information.

Test V1 is just to send a simple request for 5 carnitas tacos recipes.
"""

import requests
import json
import csv
#import pymongo

api_base = "https://api.edamam.com/search?"

search_q = input("What are you looking for? ")
cuisine_q = input("What is the cuisine style? ")
# Implement lemmatization using trained dataset on input in order to make 
# future database be less likely to have redundant entries 
# (e.g., taco vs tacos)

q = f"q={search_q}"

# Level up:
# Check a database of dishes to see if this query has been asked for already
# If not, do an API call

# Currently, just does an API call, may hit API limit if continuing with this
# version
with open("../secrets/edamam.json","r") as f:
  cred = json.load(f)

app_id = cred["id"]
app_id_s = f"&app_id=${app_id}"

app_key = cred["key"]
app_key_s = f"&app_key=${app_key}"

# Level up: call for a few recipes using limiter and make an "average version"
# of the input in order to get better results from the API call
# limiter = "&from=0&to=4"

api_call = api_base + q+ app_id_s + app_key_s #+ limiter

#print(api_call)

resp = requests.get(api_call)

if resp.status_code == 200:
    response_dict = resp.json()
    resp_dict_hits = response_dict['hits']
    count = 0
    with open(f"../write_data/{search_q}_edamam_api_return.json", "w") as f:
      json.dump(resp_dict_hits, f)

    with open(f"../write_data/{search_q}_cuisinetype.txt", "w") as t:
      t.write(cuisine_q)

    #return search_q, cuisine_q

else:
  print("Error, unable to retrieve. Server response code is: ", 
        resp.status_code)