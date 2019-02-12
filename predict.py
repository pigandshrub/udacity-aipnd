# PROGRAMMER: Daphne Cheung
# DATE UPDATED: 09/23/2018

import matplotlib; matplotlib.use('agg')
import torch
from torch import nn
from torch import optim
from PIL import Image
import argparse
import model_functions
import utilities

# Define command line parser
parser = argparse.ArgumentParser(description='Image Classifier - Part 2 - Predict')
parser.add_argument('image', type=str, help="path to single image file")
parser.add_argument('checkpoint', type=str, help="path to checkpoint file")
parser.add_argument('--top_k', action="store", type=int, default=1, dest="top_k", help="show top k most likely classes")
parser.add_argument('--category_names', action="store", type=str, default="", dest="category_names", help="use a mapping of categories to real names")
parser.add_argument('--gpu', action='store_true', default=False, dest="gpu", help="use gpu")

# Capture command line arguments
results = parser.parse_args()
image_path = results.image
checkpoint = results.checkpoint
top_k = results.top_k
category_names = results.category_names
gpu = results.gpu

# Load the provided checkpoint
model = utilities.load_checkpoint(checkpoint)              

# Get the inverse of class_to_idx
idx_to_class = {v: k for k, v in model.class_to_idx.items()}   

# Calculate top k probabilities and classes, using gpu if requested
probs, classes = utilities.predict(image_path, checkpoint, idx_to_class, top_k, gpu)  

# Get label map if provided
cat_to_name = utilities.label_mapping(category_names)

# Print image class and top k class probabilities
utilities.print_image_prob(image_path, cat_to_name, top_k, probs, classes)  
