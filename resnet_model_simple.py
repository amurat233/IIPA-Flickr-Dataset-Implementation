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

loss_fn = nn.MSELoss()

json_name = "dataset_stats.json"
device = torch.device("cuda")

wandb.init(project="SMP IIPA Pytorch")

# Defining hyperparameters and certain settings
val_percent = 0.2  # Percent of images used for validation
batch_size = 16  # batch size
lr = 0.0001  # Learning Rate
momentum = 0.5
random_seed = 99  # Don't Change. Random Seed for train_test_split.
epochs = 10
save = True
freeze = False

# Saves information about the run to wandb
config = wandb.config
config.batch_size = batch_size
config.validation_percentage = val_percent
config.lr = lr
config.random_seed = random_seed
config.freeze = freeze

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
if freeze:
    for param in model.parameters():
        param.requires_grad = False
model.fc = torch.nn.Linear(in_features=2048, out_features=1)

try:
    model.load_state_dict(
        torch.load(
            "models/resnet_model.pth",
            map_location=device))
except BaseException:
    print("MODEL NOT LOADED")
    nn.init.kaiming_uniform_(model.fc.weight)

model.to(device)
wandb.watch(model)

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


train_images = torch.Tensor(train_images)
train_labels = torch.Tensor(train_labels)

train_ds = TensorDataset(train_images, train_labels)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

val_images = torch.Tensor(val_images)
val_labels = torch.Tensor(val_labels)

val_ds = TensorDataset(val_images, val_labels)
val_dl = DataLoader(val_ds, batch_size)

@torch.no_grad() #Makes it so that gradients aren't kept track of during evaluation
def evaluate(model, val_dl):#Evaluates the model on the test_set
    global device

    model.eval()
    losses = []
    
    for batch in val_dl:
        image_nums, labels = batch
        image_nums = [int(num) for num in image_nums.tolist()]
        images = prepare_images(image_nums)
        images = centre_crop(images)
        images = images.to(device)
        labels = labels.to(device)

        labels = labels.unsqueeze(1)
        preds = model(images)
        loss = loss_fn(preds, labels).item()

        losses.append(loss)
    
    return np.mean(np.array(losses))


def fit(model, train_dl, val_dl, lr, epochs):
    global device
    global momentum

    opt = torch.optim.SGD(model.parameters(), lr, momentum = momentum)

    for epoch in range(epochs):
        training_losses = []
        # TRAINING STEP
        ##########################################################################
        model.train()
        for batch in train_dl:
            image_nums, labels = batch
            image_nums = [int(num) for num in image_nums.tolist()]
            images = prepare_images(image_nums)
            images = train_transform(images)
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.unsqueeze(1)
            preds = model(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
            training_losses.append(loss.item())
        epoch_train_loss = np.mean(np.array(training_losses))
        wandb.log({"Training Loss": epoch_train_loss, "Epoch": epoch})
        ##########################################################################

        #VALIDATION STEP
        epoch_val_loss = evaluate(model, val_dl)
        wandb.log({"Validation Loss": epoch_val_loss, "Epoch": epoch})

        print(f"Epoch {epoch +1 }/{epochs}, Training Loss:{epoch_train_loss}, Validation Loss:{epoch_val_loss}")


fit(model, train_dl, val_dl, lr, epochs)

if save:
    torch.save(model.state_dict(), "models/resnet_model.pth")