from urllib.request import urlretrieve
import time

start = time.time()

with open("train_urls.txt", "r") as f:
    urls = [url.strip("\n") for url in f.readlines()]

with open("num_downloaded.txt", "r") as f:
    already_downloaded = int(list(f.readlines())[0])
urls = urls[already_downloaded:]

def update_downloaded(new_downloaded):
    with open("num_downloaded.txt", "w") as f:
        f.write(str(new_downloaded))

def download_urls(urls, already_downloaded):
    count = 0
    for i,url in enumerate(urls):
        name = "train_images/" + str(already_downloaded + i) + ".jpg"
        urlretrieve(url, name)

        if (i+1)%20 == 0:
            update_downloaded(already_downloaded + i + 1)
        count += 1
    update_downloaded(already_downloaded + count)

download_urls(urls, already_downloaded)
print(time.time() - start)