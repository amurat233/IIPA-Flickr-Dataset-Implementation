import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt

with open("train_label.txt", "r") as f:
    labels = [float(label.strip("\n")) for label in f.readlines()]

images = list(range(len(labels)))

df = {"image": images, "label": labels}
df = pd.DataFrame(df)
df.sort_values(by='label', ascending=True,inplace = True)

group_size = len(labels)//3
lst = [df.iloc[i:i+group_size] for i in range(0,len(df)-group_size+1,group_size)]

for i,df in enumerate(lst):
    folder_name = str(i+1) + "/"
    for image in df["image"]:
        image_name = str(image) + ".jpg"
        copyfile("train_images/" + image_name, "train_images_small_groups/" + folder_name + image_name)