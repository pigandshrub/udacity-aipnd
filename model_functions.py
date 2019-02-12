# PROGRAMMER: Daphne Cheung
# DATE CREATED: 09/23/2018

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

train_batch_size = 32
test_batch_size = 16

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256), # Because the image's shortest side is 256 pixels
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# Load the dataset, define and return the dataloaders
def set_dataloaders(data_dir):
    print ("Loading ...")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=True)
    print("Finished loading data and defining dataloaders.")
    return train_dataset, train_loader, valid_loader, test_loader

# This function loads one of 3 pre-trained networks
# Load pre-trained network
def load_network(arch):
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    return model
                
# Define the structure of the network
class linear_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_p):
        ''' Builds a feedforward linear network with the following arguments:
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            dropout_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        self.input = nn.Linear(input_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_p)
                                      ])
        self.output = nn.Linear(hidden_layers[len(hidden_layers)-1], output_size)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the log softmaxes '''
        x = self.input(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    
# Define a new classifier, replace pre-trained model's classifier with 
# the one we just defined, and return the updated model
def set_classifier(arch, input_size, output_size, hidden_layers, dropout_p, model):
    print("Defining network with: ")
    print("... architecture: {}".format(arch))
    print("... input size: {}".format(input_size))
    print("... output size: {}".format(output_size))
    print("... hidden units: {}".format(hidden_layers))
    new_classifier = linear_network(input_size, output_size, hidden_layers, dropout_p)
    model.classifier = new_classifier
    print("Finished replacing pre-trained model's classifier with new network.") 
    return model

# Train and validate the model
def train_checkpoint(model, arch, input_size, output_size, hidden_layers, dropout_p, lr, epochs, gpu, train_loader, valid_loader, save_dir, train_dataset):
    print("Start training model using:")
    print("... learning rate: {}".format(lr))
    print("... epochs: {}".format(epochs))
    print("... gpu: {}".format(gpu))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    # Set device to train model
    device = "cpu"
    if (gpu):
        device = "cuda"
        
    model.to(device)

    # Start training and validating
    print_every = train_batch_size
    running_loss = 0
    steps = 0
    for e in range(epochs):
        model.train()
        for (inputs, targets) in iter(train_loader):
            steps += 1

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Disable dropout
                model.eval()

                # We don't need to calculate gradients when passing values forward during validation
                with torch.no_grad():
                    test_loss = 0
                    accuracy = 0
                    for images, imtargets in iter(valid_loader):
                        images, imtargets = images.to(device), imtargets.to(device)
                        results = model.forward(images)
                        test_loss += criterion(results, imtargets).item()

                        # Remember to take the exponential of the output because criterion outputs log softmax.
                        ps = torch.exp(results)
                        equality = (imtargets.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                running_loss = 0
                model.train()
                
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'dropout': dropout_p,
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'training': {
                      'epochs': epochs, 
                      'learning_rate': lr
                  }
                 }
    print("Saving checkpoint to directory {} ...".format(save_dir))
    torch.save(checkpoint, save_dir)  # Save in given directory
    print("Done.")
