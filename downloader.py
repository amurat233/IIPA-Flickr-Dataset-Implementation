from urllib.request import urlretrieve
import flickrapi

flickr=flickrapi.FlickrAPI('96a3893fdf81e9c1bc971722547ef8dc', 'ab04aa8f72e1a5a3',format='parsed-json')
photos = flickr.photos.getInfo(photo_id = "385070026")

class ImageDownloader:
    def __init__(self, urls_loc, api):
        self.flickr = api
        pids = self.read_ids(urls_loc)
        #Gets the number of images seen before. Needed to remove failed downloads.
        with open("num_looked_at.txt", "r") as txt_file:
            self.num_seen, self.num_downloaded = list(txt_file.readlines())[0].split(",")

        #List of indicies to remove from labels once I finish downloading
        with open("bad_indicies.txt", "r") as txt_file:
            self.bad_indicies = list(txt_file.readlines())[0].split(",")

    def get_url(self, pid, size = "n"):
        photo = self.flickr.photos.getInfo(photo_id = pid)
        secret = photo["photo"]["secret"]
        server = photo["photo"]["server"]

        url = f"https://live.staticflickr.com/{server}/{pid}_{secret}_{size}.jpg"
        return url
    
    def read_ids(self, urls_loc):
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

    def download_images(self, num_images):
        downloaded_now = 0
        if num_images == -1:
            images_to_download = len(self.pids)
        else:
            images_to_download = num_images
        
        for i in range(images_to_download):
            if (i+1)%30 == 0:
                with open("num_looked_at.txt", "w") as txt_file:
                    txt_file.write(f"{self.num_seen + i},{self.num_downloaded + downloaded_now}")
                with open("bad_indicies.txt", "a") as txt_file:
                    txt_file.write(",".join(self.bad_indicies))
                with open("train_img.txt") as txt_file:
                    txt_file.write()

            try:
                url = self.get_url(self.ids[i])
            except:
                self.bad_indicies.append(self.num_seen+1)
                continue
            name = "train_images/" + str(self.num_downloaded + downloaded_now) + ".jpg"
            urlretrieve(url, name) #Downloads the image
            downloaded_now += 1
            

downloader = ImageDownloader("train_img.txt", "num_looked_at.txt", flickr)