import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import json
import numpy as np


json_name = "dataset_stats.json"
device = torch.device("cuda")

wandb.init(project="SMP IIPA Pytorch")

# Defining hyperparameters and certain settings
val_percent = 0.2  # Percent of images used for validation
batch_size = 16  # batch size
lr = 0.01  # Learning Rate
momentum = 0.5
random_seed = 99  # Don't Change. Random Seed for train_test_split.
epochs = 50
save = True
freeze = False

# Saves information about the run to wandb
config = wandb.config
config.batch_size = batch_size
config.validation_percentage = val_percent
config.lr = lr
config.random_seed = random_seed
config.freeze = freeze

#Metrics
def get_accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
loss_fn = F.cross_entropy

with open(json_name) as file:
    stats = json.load(file)
    std = [round(i, 3) for i in stats["std"]]
    mean = [round(i, 3) for i in stats["mean"]]
    n = stats["n"]

model = torchvision.models.resnet50()
if freeze:
    for param in model.parameters():
        param.requires_grad = False
model.fc = torch.nn.Linear(in_features=2048, out_features=10)

if save:
    try:
      model.load_state_dict(
            torch.load(
                "models/resnet_classifier_model.pth",
                map_location=device))
    except BaseException:
        print("MODEL NOT LOADED")
        nn.init.kaiming_uniform_(model.fc.weight)

model.to(device)
#wandb.watch(model)

Transform = transforms.Compose([
    # Scales the short side to 256px. Aspect ratio unchanged. Then center
    # crops to make the size of all images equal
    transforms.Resize(256),
    transforms.CenterCrop([256, 256]),
    # Converts PIL Image to tensor
    transforms.ToTensor(),
    # Normalises each channel by subtracting the mean from each channel and
    # dividing by the std
    transforms.Normalize(mean, std)
])

#train_transform randomly crops and flips images, used to prevent overfitting
train_transform = transforms.Compose([transforms.RandomResizedCrop([224, 224]), transforms.RandomHorizontalFlip()])
#When evaluating the models performance you want the test images to be the same, so it centre crops instead.
centre_crop = transforms.CenterCrop([224, 224])

#Loads images from folder as an ImageFolder dataset
dataset = ImageFolder("train_images_grouped", transform = Transform)

#Randomly splits the dataset into training and validation datasets
val_size = int(len(dataset)*val_percent)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

#Creates dataloaders. They batch the images and act as iterator objects that return these batches.
train_dl = DataLoader(train_ds, batch_size, shuffle = True)
val_dl = DataLoader(val_ds, batch_size)

@torch.no_grad() #Turns of all gradient tracking when this function is run
def evaluate(model, val_dl):#Evaluates the model on the test_set
    global device

    model.eval() #Turns of dropout layers and other training functionality
    losses = [] #List to store batch losses
    accuracies = [] #List to store batch accuracies

    for batch in val_dl:
        images, labels = batch
        #Centre crops the image
        images = centre_crop(images)
        #Sends the image and data to the GPU
        images = images.to(device)
        labels = labels.to(device)

        #Feeds images into the model to get predictions, these are used to calculate the batch loss and accuracy.
        preds = model(images)
        loss = loss_fn(preds, labels).item()
        accuracy = get_accuracy(preds, labels)

        losses.append(loss)
        accuracies.append(accuracy)
    
    #Returns the average loss and accuracy for all the batches
    return np.mean(np.array(losses)),np.mean(np.array(accuracies)) 


def fit(model, train_dl, val_dl, lr, epochs):
    global device
    global momentum

    #The optimiser uses stochastic gradient descent
    opt = torch.optim.SGD(model.parameters(), lr, momentum = momentum)

    for epoch in range(epochs):
        training_losses = [] #List for storing training batch losses
        training_accuracies = [] #List for storing training batch accuracies
        # TRAINING STEP
        ##########################################################################
        model.train() #Enables training features like dropout layers
        for batch in train_dl:
            images, labels = batch
            #Prepares images and sends them to the gpu
            images = train_transform(images)
            images = images.to(device)
            labels = labels.to(device)

            #Runs the image through the model and gets the predictions
            preds = model(images)
            loss = loss_fn(preds, labels)# Gets the loss
            loss.backward() #Calculates the gradients
            opt.step() #Does an optimiser step
            opt.zero_grad() #Zeros the gradients so the don't build up

            #Stores the accuracy and loss of each batch
            training_losses.append(loss.item())
            training_accuracies.append(get_accuracy(preds, labels))
        
        #Calculates the average loss and accuracy for the epoch
        epoch_train_loss = np.mean(np.array(training_losses))
        epoch_train_accuracy = np.mean(np.array(training_accuracies))
        #Logs these metrics using Weights and Biases
        wandb.log({"Training Loss": epoch_train_loss, "Epoch": epoch})
        wandb.log({"Training Accuracy": epoch_train_accuracy, "Epoch": epoch})
        ##########################################################################

        #VALIDATION STEP
        #Evaluates the model on the validation data loader
        epoch_val_loss, epoch_val_accuracy = evaluate(model, val_dl)
        #Logs the validation metrics using Weights and Biases
        wandb.log({"Validation Loss": epoch_val_loss, "Epoch": epoch})
        wandb.log({"Validation Accuracy": epoch_val_accuracy, "Epoch": epoch})

        #Prints the epoch, training loss, validation loss, training accuracy, and validation accuracy.
        print(f"Epoch {epoch +1 }/{epochs}, Training Loss:{epoch_train_loss}, Training Accuracy:{epoch_train_accuracy}, Validation Loss:{epoch_val_loss}, Validation Accuracy:{epoch_val_accuracy}")
        #Saves the model after every epoch in case it starts overfitting.
        torch.save(model.state_dict(), "models/resnet_classifier_model" + str(epoch+1) + ".pth")


# fit(model, train_dl, val_dl, lr, epochs)

