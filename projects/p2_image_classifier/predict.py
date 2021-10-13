import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import json

from predictor import predict

parser = argparse.ArgumentParser(description='Predict flower image')

parser.add_argument('image_path', action = "store")
parser.add_argument('saved_model', action = "store")
parser.add_argument('--top_k', action = "store", dest = "top_k", type = int)
parser.add_argument('--category_names', action = "store", dest = "category_names")

print(parser.parse_args())
result = parser.parse_args()

image_path = result.image_path
saved_model = result.saved_model
top_k = result.top_k
category_names = result.category_names

if top_k == None:
    top_k = 3

# call saved model

reloaded_model = tf.keras.models.load_model(saved_model, custom_objects = {"KerasLayer" :hub.KerasLayer})


# predict image
proba, classes = predict(image_path, reloaded_model, top_k)

if category_names != None:
    with open(category_names, 'r') as cat_name:
        class_names = json.load(cat_name)
    keys = [str(i+1) for i in classes]
    names = [class_names.get(key) for key in keys]
    print

# print result
print("\nPredicted top class names: {}".format(top_k))
for i in range(top_k):
    print("\nClass name: {}".format(names[i]), "\nProbability: {:0.2%}".format(proba[i]))
