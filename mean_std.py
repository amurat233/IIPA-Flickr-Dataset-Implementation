import json
from PIL import Image
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


def prepare_image(image):
    image_name = image + ".jpg"
    image = Image.open("train_images/" +image_name).resize([256, 256], Image.BILINEAR)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = np.array(image) / 255
    #Sets RGB channels as first dimension making it easier to seperate
    #the channels for each image when doing the calculations.
    image = np.transpose(image, [2, 0, 1])
    return image

def combine(stats_1, stats_2):
    sd_1, m_1, n_1 = [np.array(i) for i in stats_1.values()]
    sd_2, m_2, n_2 = [np.array(i) for i in stats_2.values()]

    # Calculate sum x^2 for each set of data
    sx2_1 = (sd_1**2) * (n_1 - 1) + n_1 * (m_1**2)
    sx2_2 = (sd_2**2) * (n_2 - 1) + n_2 * (m_2**2)
    sx2_tot = sx2_1 + sx2_2

    # Calculates the total mean
    m_tot = (n_1 * m_1 + n_2 * m_2) / (n_1 + n_2)

    # Calculate total S_{xx}
    sxx_tot = sx2_tot - (n_1 + n_2) * (m_tot**2)

    # Combined standard deviation:
    sd_tot = np.sqrt(sxx_tot / (n_1 + n_2 - 1))

    return {"std": list(sd_tot), "mean": list(m_tot), "n": int(n_1 + n_2)}

#Updates the json file with the newly calculated stats
def update_json(stats_json, new_stats):
    global json_name
    #Combines the old stats with new stats
    stats = combine(stats_json, new_stats)
    #Overwrites the old json file with new stats
    with open(json_name, "w") as outfile:
        json.dump(stats, outfile)

def get_new_stats(json_name):
    while True:
        # Opens the nescessary files
        ######################################################################
        with open(json_name) as file:
            stats_old = json.load(file)
        ######################################################################

        images_list = []  # Initialises list in which numpy arrays representing images are stored
        count = 0
        #Looks at the images in batches of 100 so that little progress is lost if the program
        #crashes
        for i in range(100):
            seen = stats_old["n"] #Stores how many images have already been seen
            image_name = str(seen + i) #Stores the image name
            try:
                image = prepare_image(image_name) #Opens the image
            except:
                break #Breaks the loop once it can't find any more images
            count += 1
            images_list.append(image)
        # concatenates images in image_list and calculates stats
        com_images = np.concatenate(images_list, axis=1)
        sd = [com_images[i].std(ddof=1) for i in range(3)] #Standard deviation
        m = [com_images[i].mean() for i in range(3)] #Mean
        n = count #Count, needed when combining stats
        stats_new = {"std": sd, "mean": m, "n": n}
        # Updates the json using the new calculated stats
        update_json(stats_old, stats_new)

json_name = "dataset_stats.json"

get_new_stats(json_name)
