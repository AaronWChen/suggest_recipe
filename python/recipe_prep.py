""" This Python script handles data importation and preparation from the 
"What's Cooking?" Kaggle challenge: https://www.kaggle.com/c/whats-cooking

This script should take in the given json files (train.json and test.json).
train.json will be split in to training and test sets while test.json will be 
treated as a validation set. The output should be tokenized vectors that can be
used in a neural network.
"""
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
import string
import ast
import gensim 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from operator import itemgetter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

with open('../raw_data/train.json') as json_file:
  train_file = json.load(json_file)

with open('../raw_data/test.json') as json_file:
  val_file = json.open("../raw_data/test.json")
  
save_path = "../write_data/"

# Using some stopwords from https://github.com/AlludedCrabb/sound-tasty
cooking_stop_words = list(set([
        'canned', 'cans', 'drained', 'and', 'halved', 'cup', 'cups',
        'teaspoon', 'tablespoon', 'teaspoons', 'tablespoons',
        'finely', 'freshly', 'fresh', 'thickcut', 'to', 'taste',
        'grated', 'cut', 'into', 'wedges', 'pounds', 'unpeeled', 'large',
        'minced', 'slice', 'slices', 'sliced', 'thick-cut', 'cut',
        'crosswise', 'pieces', 'toothpicks', 'low-fat', 'chopped', 'or',
        'taste', 'cooked', 'dry', 'shredded', 'beaten', 'dried', 'melted',
        'stems', 'removed', 'diced', 'ounce', 'ounces', 'packages',
        'softened', 'such', 'RedHot®', 'RedHot', 'Franks', "Frank's",
        'crumbled', 'Old', 'Bay®', 'Bay', 'pinch', 'for', 'garnish', 'slice',
        'slices', 'needed', 'inch', 'cubes', 'cooking', 'spray', 'ground',
        'rotisserie', 'lowfat', 'as', 'quarteres', 'cloves', 'more', 'can',
        'package', 'frozen', 'thawed', 'packet', 'reducedfat', 'Knorr',
        'container', 'pound', 'peeled', 'deveined', 'seeded', 'ripe',
        'English', 'juiced', 'plus', 'more', 'Hass', 'cubed', 'Mexicanstyle',
        'hearts', 'prepared', 'party', 'pitted', 'mashed',
        'roma', 'optional', 'chunk', 'Hot', 'bunch', 'cleaned', 'box',
        'chickenflavored', 'Golden', 'delicious', 'cored', 'any', 'flavor',
        'flavored', 'whole', 'allpurpose', 'all', 'purpose', 'deep', 'frying',
        'dash', 'packed', 'in', 'French', 'jar', 'small', 'head', 'little',
        'smokie', 'seasoned', 'Boston', 'Bibb', 'leaves', 'lean', 'pickled',
        'Asian', 'dark', 'flaked', 'rolled', 'packed', 'jellied',
        'thirds', 'with', 'attached', 'skewers', 'skinless', 'boneless',
        'half', 'kernels', 'rinsed', 'quart', 'quarts', 'kernel',
        'Italianstyle', 'unpopped', 'lightly', 'coating', 'SAUCE',
        'lengthwise', 'miniature', 'semisweet', 'rinsed', 'round',
        'squeezed', 'stewed', 'raw', 'the', 'liquid', 'reserved', 'medium',
        'instant', 'solid', 'pack', 'refrigerated', 'halves', 'distilled',
        'loaf', 'extra', 'virgin', 'crushed', 'kosher', 'toasted', 'buttery',
        'TM', 'panko', 'Japanese', 'regular', 'bottle', 'bottles', 'thin',
        'peel', 'paper', 'thick', 'circles', 'unbleached',
        'breast', 'breasts', 'wings', 'strips', 'jumbo', 'giant', 'chunks',
        'quickcooking', 'sweetened', 'flakes', 'Ranchstyle', 'snipped',
        'food', 'ROTEL', 'Italian', 'sticks', 'stick', 'crescent', 'thinly',
        'boiled', 'Genoa', 'roasted', 'thin', 'extrasharp', 'pressed',
        'sifted', 'split', 'tips', 'discarded', 'mini', 'deli', 'drain',
        'reserve', 'diameter', 'Greek', 'Thai', 'drops', 'square', 'crusty',
        'American', 'selfrising', 'imitation', 'Wings', 'apart', 'at',
        'joints', 'wing', 'tips', 'discarded', 'parts',
        'tops', 'seperated', 'blend', 'coarsely', 'sweet', 'stalk', 'heads',
        'husked', 'divided', 'pats', 'unsalted', 'active', 'warm', 'sea',
        'separated', 'herb', 'overripe', 'degrees', 'F', 'C', 'room',
        'temperature', 'machine', 'very', 'pint', 'puree', 'coarse',
        'envelopes', 'lukewarm', 'creamstyle', 'unsweetened',
        'lite', 'of', 'chilled', 'freezer', 'cold', 'brushing', 'nonfat',
        'squares', 'tails', 'thigh', 'quarters', 'Masterpiece', 'KC', 'from',
        'El', 'Paso', 'bulk', 'Hunts', 'Roma', 'light', 'fluid', 'lagerstyle',
        'stalks', 'quartered', 'undrained', 'drained', 'Tony', 'Chacheres',
        'lump', 'uncooked', 'cube', 'bits', 'hair', 'angel', 'trimmed',
        'stew', 'spaghetti', 'brisket', 'bitesized', 'matchstick', 'Chobani',
        'unbaked', 'crust', 'torn', 'bonein', 'pounded', 'bitesize',
        'granules', 'boiling', 'yolk', 'coloring', 'pinch', 'a', 'blender',
        'fine', 'which', 'extralarge', 'use', 'will', 'make', 'garnish',
        'barely', 'moistened', 'about', 'right', 'before', 'serving', 'mix',    
    ]))

unhelpful = list(set(['fresh', 'ripe', 'cracked', 'cooking', 'coarse', 'light', 
             'mild', 'hot', 'minced', 'dark roast', 'unsifted', 'canned', 
             'cans', 'drained', 'halved', 'finely', 'freshly', 'thickcut', 
             'grated', 'cut', 'unpeeled', 'large', 'minced', 'slice', 
             'slices', 'sliced', 'chopped','shredded', 'beaten', 'dried', 
             'melted', 'stems', 'softened', 'packages', 'crumbled', 'ground',
             'low-fat', 'rotisserie', 'lowfat', 'can', 'thawed', 'packet', 
             'reducedfat', 'small', 'pats', 'regular', 'lukewarm', 'mashed', 
             'stalk', 'breast', 'breasts', 'juiced', 'halves', 'extrasharp', 
             'sharp', 'extra sharp', 'frozen', 'raw', 'warm', 'divided', 
             'little', 'squares', 'thinly', 'thick', 'rinsed', 'toasted', 
             'bitesize', 'chunks', 'refrigerated', 'kernel', 'kernels', 
             'jar', 'lengthwise', 'unpeeled', 'cleaned', 'paper', 'melted', 
             'separated', 'seperated', 'deveined', 'party', 'bunch', 'overripe', 
             'boiled', 'chunk', 'container', 'bitesized', 'sweet', 'strips', 
             'sifted', 'roma', 'very', 'undrained', 'stewed', 'thawed', 'lean', 
             'roasted', 'extra', 'lite', 'coarsely', 'pressed', 'square', 
             'jumbo', 'yolk', 'yolks', 'barely', 'pitted', 'cored', 'puree', 
             'cubes', 'angel', 'hair', 'angelhair', 'giant', 'husked', 'chilled', 
             'thigh', 'trimmed', 'thin', 'lightly', 'cubed', 'drops', 'grated', 
             'boneless', 'unsalted', 'pieces', 'skinless', 'pounded', 
             'chickenflavored', 'extralarge', 'medium', 'reserve', 'unbaked', 
             'crushed', 'wings', 'crosswise', 'cold', 'bonein', 'bone in', 
             'squeezed', 'kosher', 'miniature', 'tails', 'quarters', 'attached', 
             'loaf', 'dry', 'more', 'head', 'removed', 'packed', 'hearts', 
             'matchstick', 'unbleached', 'heads', 'stems', 'sea', 'diced', 
             'mini', 'cut', 'unpopped', 'box', 'uncooked', 'freezer', 'stalks', 
             'shredded', 'halved', 'snipped', 'thick-cut', 'split', 'seeded', 
             'sweetened', 'discarded', 'lump', 'boiling', 'whole', 'semisweet', 
             'semi-sweet', 'quartered', 'moistened', 'reserved', 'prepared', 
             'fresh', 'ripe', 'cracked', 'cooking', 'coarse', 'light', 'mild', 
             'hot', 'minced', 'dark roast', 'unsifted', 'quaker', 'raw', 'frozen', 
             'calore-wise', 'ziploc bag', 'real', 'lite', 'crisp', 'decaffeinated', 
             'canned', 'processed', 'cooked', 'unpeeled', ]))

brands = ['rotel', 'absolut', 'betty crocker', 'jello', 'diana', 'ener-g', 
          'del-monte', "hunt's", 'martha', 'goya', 'cracker barrel', 
          'hamburger helper', "mccormick's", 'pepperidge farm', 'knorr', 
          'godiva', 'hidden valley', 'tabasco', 'branston', "kellogg's", 
          'hodgson mill', 'kraft', 'johnsonville', 'jim beam', 'mccormick', 
          'equal', 'jell-o', 'jimmy dean', 'country bob', "smucker's", 
          'toblerone', 'gerber', 'nestle', 'nestl', 'malt-o-meal', 'triscuit', 
          'ragu', 'campbell', 'hormel', 'earth balance', 'pillsbury', 
          "bird's eye", "campbell's", "betty crocker's", 'gold medal', 
          'crystal light', 'milnot', "land o' lakes", 'herb-ox', 'quaker',
          'coffee-mate', 'contadina', 'j&d', 'fantastic foods', 'bacardi', 
          'eckrich', 'little smokies', 'snickers', 'ortega', 'bayou blast', 
          "annie's", 'mrs. dash', 'mori-nu', 'old el paso', 'original supreme',
          'morton', 'nabisco', 'rice-a-roni', 'stolichnaya', "lawry's", 
          'st. germain', "eggland's best", 'club house "lagrille"', 'hostess',
          'giada de laurentiis genovese', '*available in most target stores', 
          'jarlsberg', 'pillsbury plus', 'ro-tel', 'pillsbury grands', 
          'shilling', 'hershey', 'hershey carb alternatives', 'pasta roni', 
          'pastaroni', 'torani', 'v8', 'v8 fusion', 'ghiradelli', 'oscar mayer',
          "bird's", 'smithfield', 'cadbury', 'sun-maid', 'karo', 
          'wishbone deluxe', 'vochelle', 'laughing cow', 'omega', 'stirrings',
          'duncan hines', 'barilla',
         ]

stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += unhelpful
stopwords_list += brands

targets = [target['cuisine'] for target in train_file]
X = []
X = train_file.drop('cuisine', axis=1)
