import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt

json_name = "dataset_stats.json"
device = torch.device("cuda")
image_name = "412.jpg"
class_mapping = {'1': 0, '10': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
id_to_class = inv_map = {str(v): int(k) for k, v in class_mapping.items()}
print(id_to_class)


with open(json_name) as file:
    stats = json.load(file)
    std = [round(i, 3) for i in stats["std"]]
    mean = [round(i, 3) for i in stats["mean"]]
    n = stats["n"]

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
        transforms.Resize([224,224]),
        transforms.CenterCrop([224, 224]),
        # Converts PIL Image to tensor
        transforms.ToTensor(),
        # Normalises each channel by subtracting the mean from each channel and
        # dividing by the std
        transforms.Normalize(mean, std)
    ])
    image = Transform(image)
    image = image.unsqueeze(0)
    return image

with torch.no_grad():
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(in_features=2048, out_features=10)
    model.load_state_dict(torch.load("models/resnet_classifier_model50.pth", map_location=device))
    model.to(device)

    image = Image.open("train_images/"+image_name)
    image = prepare_image(image)
    plt.imshow(image[0].permute(1,2,0))
    plt.show()
    image = image.to(device)

    pred = model(image)
    print(pred.tolist())
    _, idx = torch.max(pred, dim = 1)
    idx = str(idx.item())
print(id_to_class[idx])