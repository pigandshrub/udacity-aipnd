# PROGRAMMER: Daphne Cheung
# DATE UPDATED: 09/23/2018

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

import model_functions

# Load label mapping if available, else return None
def label_mapping(category_names):
    import json
    if (category_names != ""):    
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
    else:
        return None
    
# Load checkpoint
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = model_functions.load_network(checkpoint['arch'])
    # Rebuild the classifier
    new_classifier = model_functions.linear_network(checkpoint['input_size'],
                              checkpoint['output_size'],
                              checkpoint['hidden_layers'],
                              checkpoint['dropout'])
    
    model.classifier = new_classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Process the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize so that the shortest side is 256 but we keep the aspect ratio
    x, y = image.size
    resize_ratio = max(256 / x, 256 / y)
    image.thumbnail((int(x * resize_ratio), int(y * resize_ratio))) 
    
    # Crop out center 224 x 224
    # We note that PIL uses Cartesian pixel coordinate system
    # with (0,0) in the upper left corner. 
    new_x, new_y = image.size
    left = (new_x - 224)*0.5
    upper = (new_y - 224)*0.5
    right = new_x - (new_x - 224)*0.5
    lower = new_y - (new_y - 224)*0.5
    image = image.crop((left, upper, right, lower))

    # Convert color channel values to float 0 - 1, subtract the means, 
    # divide the standard deviations
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image)
    # Take original channel values as a fraction of their max value of 255
    np_image = np_image / 255  
    np_image = (np_image - mean) / std
    
    # Reorder color channel so that it is the first dimension
    np_image = np_image.transpose((2, 0, 1))

    # Return the tensor equivalent of the Numpy array
    return torch.from_numpy(np_image)

# Display the image
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Convert tensor back to Numpy array
    np_image = np.array(image)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    np_image = np_image.transpose(1, 2, 0)
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = std * np_image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    np_image = np.clip(np_image, 0, 1)
    
    ax.imshow(np_image)
  
    return ax

def predict(image_path, checkpoint_path, idx_to_class, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    
    # Load model checkpoint
    model = load_checkpoint(checkpoint_path)  
    # Format image
    image = process_image(image)               
    device = "cpu"
    if (gpu):
        device = "cuda"
    model.to(device)
    image = image.to(device)
    
    # Get results from model, adding batch size of 1 to tensor 
    # and make sure tensor is float tensor.
    results = model.forward(image.unsqueeze(0).float()) 
    # Get the probabilities of the results
    ps = torch.exp(results) / torch.sum(torch.exp(results)) 
    # Get the top k probabilities and their idx
    probs, idx = torch.topk(ps, top_k)                             
    probs = [float(p) for p in probs[0]]
    classes = [idx_to_class[x] for x in np.array(idx[0])]
    return probs, classes


# Print image class and top k class probabilities
def print_image_prob(image_path, cat_to_name, top_k, probs, classes): 
    name = image_path.split("/")[-2]
    if (cat_to_name != None):
        name = cat_to_name[name]
        
    # Display top k flower classes/names
    if (cat_to_name != None):
        names = [cat_to_name[classes[i]] for i in range(len(classes))]
    else:
        names = classes
        
    # Place data into a dataframe
    data = {'Probability' : probs,'Class' : names }  
    flowers = pd.DataFrame(data)
    print("")
    print("Target Class: {}".format(name))
    print("--- List of Top k Classes (k = {}) ---".format(top_k))
    print(flowers)
    print("")
