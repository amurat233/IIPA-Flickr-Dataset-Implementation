import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import json


wandb.init(project="Flickr IIPA Pytorch")

# Defining hyperparameters and certain settings
test_run = False  # Set to False during actual training
test_run_size = 1024  # Number of image pairs used in a test run
training_experiment = False
val_percent = 0.2  # Percent of images used for validation
batch_size = 16  # batch size
lr = 0.05  # Learning Rate
random_seed = 99  # Don't Change. Random Seed for train_test_split.
momentum = 0.9  # If using SGD.
epochs = 20
loss_fn = nn.MSELoss()

# Saves information about the run to wandb
# config = wandb.config
# config.batch_size = batch_size
# config.validation_percentage = val_percent
# config.lr = lr
# config.random_seed = random_seed
# config.momentum = momentum

json_name = "dataset_stats.json"
device = torch.device("cuda")

with open(json_name) as file:
    stats = json.load(file)
    std = [round(i, 3) for i in stats["std"]]
    mean = [round(i, 3) for i in stats["mean"]]
    n = stats["n"]

with open("train_label.txt", "r") as f:
    labels = [float(label.strip("\n")) for label in f.readlines()]

def prepare_image(image):
    global mean
    global std
    # Converts image to RGB
    if image.mode != 'RGB':
        image = image.convert("RGB")
    # Adds necessary transforms to image
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
    image = Transform(image)
    return image

# Given a batch of images, it applies prepare_image() on each and returns
# a tensor of image pairs
def prepare_images(image_numbers):
    #unpop_images = [Image.open("image_dataset/"+str(i[1][0])+".jpg") for i in images.iterrows()]
    image_tensor = torch.stack([prepare_image(Image.open("train_images/" + str(number) + ".jpg")) for number in image_numbers])
    return image_tensor

train_transform = transforms.Compose([transforms.RandomResizedCrop([224, 224]), transforms.RandomHorizontalFlip()])
centre_crop = transforms.CenterCrop([224, 224])

class IIPAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == "resnet":
            self.model = torchvision.models.resnet50()
            self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
            nn.init.kaiming_uniform_(self.model.fc.weight)
        else:
            self.model = EfficientNet.from_pretrained(model_name, num_classes = 1)
            nn.init.kaiming_uniform_(self.model._fc.weight)

    def forward(self, batch):
        preds = self.model(batch)
        return preds


images = list(range(len(labels)))

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size = val_percent, random_state=random_seed, shuffle = True)


@torch.no_grad() #Makes it so that gradients aren't kept track of during evaluation
def evaluate(model, image_numbers, labels, batch_size):#Evaluates the model on the test_set
    global device

    model.eval()
    results = []
    if len(image_numbers) % batch_size != 0:
        num_batches = (len(image_numbers) // batch_size) + 1
    else:
        num_batches = (len(image_numbers) // batch_size)

    for i in range(num_batches):
        batch = prepare_images(image_numbers[batch_size*i:batch_size*(i+1)])
        batch = centre_crop(batch)
        batch = batch.to(device)
        batch_labels = torch.Tensor(labels[batch_size*i:batch_size*(i+1)])
        batch_labels = batch_labels.to(device)
        
        preds = model(batch)[:,0]

        loss = loss_fn(preds, batch_labels).item()
        results.append(loss) #Stores the batch losses and accuracies in results
    return sum(results)/len(results)

model = IIPAModel("resnet")
model.to(device)

def fit(model, train_numbers, val_numbers, train_labels, val_labels, batch_size, lr, epochs):
    global device
    global momentum

    # TRAINING STEP
    ##########################################################################
    if len(val_numbers) % batch_size != 0:
        num_batches = (len(val_numbers) // batch_size) + 1
    else:
        num_batches = (len(val_numbers) // batch_size)

    for epoch in range(epochs):
        train_losses = []
        model.train()
        opt = torch.optim.SGD(model.model.parameters(), lr, momentum=momentum)
        for i in range(num_batches):
            batch = prepare_images(train_numbers[batch_size * i:batch_size * (i + 1)])
            # Randomly crops images before feeding into model. Reduced
            # overfitting and has a similiar effect to getting more data.
            batch = train_transform(batch)
            batch = batch.to(device)
            batch_labels = torch.Tensor(train_labels[batch_size*i:batch_size*(i+1)])
            batch_labels = batch_labels.to(device)

            preds = model(batch)[:,0]

            loss = loss_fn(preds, batch_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.item())
        # Average training loss
        epoch_train_loss = sum(train_losses) / len(train_losses)  
        wandb.log({"Training Loss": epoch_train_loss, "Epoch": epoch})
    ##########################################################################
    # TESTING/VALIDATION STEP
    ##########################################################################
        loss = evaluate(model, val_numbers, val_labels, batch_size)
        wandb.log({"Validation Loss": loss, "Epoch": epoch})
    ##########################################################################

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {round(epoch_train_loss,3)}, Validation Loss: {round(loss,3)}")

fit(model, train_images, val_images, train_labels, val_labels, batch_size, lr, 5)