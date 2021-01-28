from urllib.request import urlretrieve
import flickrapi

flickr = flickrapi.FlickrAPI('96a3893fdf81e9c1bc971722547ef8dc', 'ab04aa8f72e1a5a3',format='parsed-json')
bad_indicies = []
urls = []

with open("train_label_full.txt", "r") as f:
    labels = [line.strip("\n") for line in f.readlines()]


def get_url(pid, size = "n"):
        photo = flickr.photos.getInfo(photo_id = pid)
        secret = photo["photo"]["secret"]
        server = photo["photo"]["server"]

        url = f"https://live.staticflickr.com/{server}/{pid}_{secret}_{size}.jpg"
        return url

def read_ids(urls_loc = "train_img.txt"):
    ids = []
    with open(urls_loc, "r") as txt_file:
        lines = txt_file.readlines()
        lines = [line[:-1] for line in lines] #Gets rid of new line character
    for line in lines:
        line = line[::-1] #reverses the line
        end = line.index("/") #finds the first /
        pid = line[:end] #Gets the reversed photo id
        ids.append(pid[::-1]) #Adds the unreversed photo id to ids
    return ids

pids = read_ids()

downloaded = 0
broken = 0
for i, pid in enumerate(pids[:100]):
    try:
        url = get_url(pid)
        print(url)
    except:
        bad_indicies.append(i)
        broken += 1
        continue
    else:
        urls.append(url)
        downloaded += 1
print(f"{round((broken/(broken+downloaded))*100,1)}% of the images are broken")

with open("train_label.txt", "w") as f:
    for i, label in enumerate(labels):
        if i not in bad_indicies:
            f.write(label + "\n")

with open("train_urls.txt", "w") as f:
    for url in urls:
        f.write(url + "\n")