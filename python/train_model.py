""" Train a multi-class classification problem.

Use the embedding for each ingredient in a recipe to suggest and predict other
recipes a user will like. They suggested recipes should not be in the same 
cuisine category.

Debug the model with a held-out validation set.

This model is heavily based on the work of Jaan Altosaar: github.com/altosaar
and the project food2vec: github.com/altosaar/food2vec.

The data for this version of the project, as well as the forked version of 
Altosar's work, is provded by Yummly from the Kaggle competition "What's 
Cooking?": https://www.kaggle.com/c/whats-cooking

Future data sources will be cited here.
"""

import tensorflow as tf
import collections
import numpy as np

layers = tf.contrib.layers

train_path = "../raw_data/train.json"
save_path = "../write_data"
embedding_size = 100
epochs_desired = 15
learning_rate = 0.025
regularization = 0.01
