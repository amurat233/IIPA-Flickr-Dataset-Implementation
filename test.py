from urllib.request import urlretrieve
import flickrapi

flickr=flickrapi.FlickrAPI('96a3893fdf81e9c1bc971722547ef8dc', 'ab04aa8f72e1a5a3',format='parsed-json')
photos = flickr.photos.getInfo(photo_id = "385070026")
print(photos)

class ImageDownloader:
    def __init__(self, urls, api):
        self.flickr = api
        self.ids = photo_ids
    
    def get_url(self, id, size = "n"):
        photo = self.flickr.photos.getInfo(photo_id = id)
        secret = photo["photo"]["secret"]
