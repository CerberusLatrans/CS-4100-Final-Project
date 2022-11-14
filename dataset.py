
import requests
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import random
import math
import urllib
import cv2 as cv
import numpy as np
from io import BytesIO
from PIL import Image

API_KEY = "AIzaSyATvdjLWeqJ0svqAXNYmdAmVU7tkaz8434" # Prob want this as a google secret through GCP
URL = "https://maps.googleapis.com/maps/api/staticmap?"
SIZE = "640x640"
MAP_ID = "5fd53f1f4a5c6512" # For styling the map -- clean map
MAP_TYPE = "satellite" # for dirty map
# center = "42.3398106,-71.0913604" <- northeastern university
# labelsTextOff="feature:all|element:labels.text|visibility: off"
# labelsIconOff="feature:all|element:labels.icon|visibility: off"

def get_map(center, zoom, type, display=False):
    full_url = URL + "center=" + center + "&zoom=" + str(zoom) + "&size=" + SIZE + "&key=" + API_KEY
    if (type == 'clean'):
        full_url += "&map_id=" + MAP_ID 
    if (type == 'dirty'):
        full_url += "&maptype=" + MAP_TYPE
    
    r = requests.get(full_url)
    img = np.array(Image.open(BytesIO(r.content)))

    if display:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.show()

    return img


def apply_canny(img, type, display=False):
    """Applies canny filter to image"""
    
    if type == 'dirty':
        edges = cv.Canny(img, 200, 400)
    elif type == 'clean':
        edges = cv.Canny(img, 5, 10)
    else:
        raise Exception("type should be either 'clean' or 'dirty'")
    
    #bw, _ = cv.threshold(edges, 200, 255, cv.THRESH_BINARY)
    #bw = cv.cvtColor(edges, )
    if display:
        plt.imshow(edges)
        plt.show()

    return edges

def upload_blob(source_file_name): #TODO
    pass
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)

    # blob.upload_from_filename(source_file_name)

    # print(
    #     f"File {source_file_name} uploaded to {destination_blob_name}."
    # )

def get_coords(location, n, radius):
    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(location) +'?format=json'
    response = requests.get(url).json()
    anchor_lat = float(response[0]["lat"]) #y
    anchor_lon = float(response[0]["lon"]) #x

    #convert miles to lat degrees
    r = radius / 69

    coordinates = []

    for _ in range(n):
        #latitude can be +- radius from the anchor lat
        lat = anchor_lat + random.uniform(-r, r)

        # from circle eqn
        delta_lon = math.sqrt(r**2 - (lat - anchor_lat)**2)
        lon = anchor_lon + random.uniform(-delta_lon, delta_lon)

        zoom = random.randint(17, 20)
        coordinates.append((lat, lon, zoom))

    # print(coordinates)
    return coordinates

if __name__ == "__main__":
    locations = ["Boston", "New York City"]
    coords = []
    for l in locations:
        coords.extend(get_coords(l, n=10, radius=1))

    temp_limit = 1
    count = 1
    for c in coords:
        center = str(c[0]) + "," + str(c[1])
        zoom = c[2]

        """ Clean Map """
        clean_image = get_map(center, zoom, "clean", display=True)
        canny_clean_image = apply_canny(clean_image, "clean", display=True)
        upload_blob(canny_clean_image)

        """ Dirty Map """
        dirty_image = get_map(center, zoom, "dirty", display=True)
        canny_dirty_image = apply_canny(dirty_image, "dirty", display=True)
        upload_blob(canny_dirty_image)

        count = count + 1
        if (count > temp_limit):
            break