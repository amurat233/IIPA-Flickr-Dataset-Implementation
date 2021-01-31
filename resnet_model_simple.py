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

json_name = "dataset_stats.json"
device = torch.device("cuda")

with open(json_name) as file:
    stats = json.load(file)
    std = [round(i, 3) for i in stats["std"]]
    mean = [round(i, 3) for i in stats["mean"]]
    n = stats["n"]

with open("train_label.txt", "r") as f:
    labels = [float(label.strip("\n")) for label in f.readlines()]

images = list(range(len(labels)))

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size = val_percent, random_state=random_seed, shuffle = True)

model = torchvision.models.resnet50()
model.fc = torch.nn.Linear(in_features=2048, out_features=1)
nn.init.kaiming_uniform_(model.fc.weight)

def prepare_image(image):
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
def prepare_images(images):
    #unpop_images = [Image.open("image_dataset/"+str(i[1][0])+".jpg") for i in images.iterrows()]
    unpop_image_tensor = torch.stack([prepare_image(Image.open(
        "image_dataset/" + str(i[1][0]) + ".jpg")) for i in images.iterrows()])
    # del unpop_images #Deletes images to save memory

    #pop_images = [Image.open("image_dataset/"+str(i[1][1])+".jpg") for i in images.iterrows()]
    pop_image_tensor = torch.stack([prepare_image(Image.open(
        "image_dataset/" + str(i[1][1]) + ".jpg")) for i in images.iterrows()])
    #del pop_images

    return torch.stack([unpop_image_tensor, pop_image_tensor], axis=1)


# Selects a random 224x224 sample of the image, decreases overfitting
train_transform = transforms.Compose(
    [transforms.RandomResizedCrop([224, 224]), transforms.RandomHorizontalFlip()])
centre_crop = transforms.CenterCrop([224, 224])


train_images = torch.Tensor(train_images)
train_labels = torch.Tensor(train_labels)

train_ds = TensorDataset(train_images, train_labels)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

for batch in train_dl:
    image_nums, label = batch
    image_nums = image_nums.tolist()
    break