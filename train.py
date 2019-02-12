# PROGRAMMER: Daphne Cheung
# DATE UPDATED: 09/23/2018

import matplotlib; matplotlib.use('agg')
import torch
from torch import nn
from torch import optim
import argparse
import model_functions
import utilities

# Define command line parser
parser = argparse.ArgumentParser(description='Image Classifier - Part 2 - Train')
parser.add_argument('data_dir', type=str, help="provide the data directory")
parser.add_argument('--save_dir', action="store", default="/home/checkpoint.pth", dest="save_dir", help="provide directory to save checkpoints")
parser.add_argument('--arch', action="store", choices={"vgg13", "vgg16", "vgg19", "alexnet"}, default="vgg16", dest="arch", type=str, help="provide architecture vgg13, vgg16, vgg19, or alexnet")
parser.add_argument('--learning_rate', action="store", default=0.0001, dest="lr", type=float, help="provide learning rate")
parser.add_argument('--hidden_units', nargs='*', default=[8192, 5096], dest="hidden_layers", type=int, help="provide hidden units, multiple values can be provided in a list")
parser.add_argument('--epochs', action="store", default=5, dest="epochs", type=int, help="provide number of epochs")
parser.add_argument('--gpu', action='store_true', default=False, dest="gpu", help="use this option to use gpu")

# Capture command line arguments
results = parser.parse_args()
arch = results.arch
data_dir = results.data_dir
save_dir = results.save_dir
gpu = results.gpu

# Define the dataloaders
train_dataset, train_loader, valid_loader, test_loader = model_functions.set_dataloaders(data_dir)

# Load pretrained model
model = model_functions.load_network(arch)

# Freeze the parameters because we're not training the features part of the model.
for param in model.parameters():
    param.requires_grad = False

# Parameters
input_size = model.classifier[0].in_features
output_size = 102

# Allow for one or more hidden units
if not isinstance(results.hidden_layers,(list,)):
    results.hidden_layers = [model.classifier[0].out_features]
hidden_layers = results.hidden_layers

dropout_p = 0.5
lr = results.lr
epochs = results.epochs

# Build new classifier and update the model with it
model = model_functions.set_classifier(arch, input_size, output_size, hidden_layers, dropout_p, model)

# Train model and save checkpoint
model_functions.train_checkpoint(model, arch, input_size, output_size, hidden_layers, dropout_p, lr, epochs, gpu, train_loader, valid_loader, save_dir, train_dataset)
