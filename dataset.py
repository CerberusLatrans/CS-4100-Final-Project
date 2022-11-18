
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
# Install Google Libraries
from google.cloud import secretmanager
from google.cloud import storage
import os

os.environ.setdefault("GCLOUD_PROJECT", "careful-striker-367620")

URL = "https://maps.googleapis.com/maps/api/staticmap?"
SIZE = "640x640"
MAP_ID = "5fd53f1f4a5c6512" # For styling the map -- clean map
MAP_TYPE = "satellite" # for dirty map
PROJECT_ID = "289867676635"
SECRET_KEY = "MAP_API_KEY"
# center = "42.3398106,-71.0913604" <- northeastern university
# labelsTextOff="feature:all|element:labels.text|visibility: off"
# labelsIconOff="feature:all|element:labels.icon|visibility: off"

def get_api_key():
    client = secretmanager.SecretManagerServiceClient()
    secret_detail = f"projects/{PROJECT_ID}/secrets/{SECRET_KEY}/versions/1"
    response = client.access_secret_version(request={"name": secret_detail})
    data = response.payload.data.decode("UTF-8")
    return data

def get_map(center, zoom, type, display=False):
    API_KEY = get_api_key()
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


def apply_canny(img, type, id, display=False):
    """Applies canny filter to image"""
    img = cv.medianBlur(img,5)
    if type == 'dirty':

        ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        kernel = np.ones((5,5),np.float32)/25
        dst = cv.filter2D(img,-1,kernel)
        edges = cv.Canny(th2, 50, 250)
        
        edges2 = cv.Canny(img, 200, 400)
    elif type == 'clean':
        edges = cv.Canny(img, 5, 10)
    else:
        raise Exception("type should be either 'clean' or 'dirty'")
    
    #bw, _ = cv.threshold(edges, 200, 255, cv.THRESH_BINARY)
    #bw = cv.cvtColor(edges, )
    if display:
        plt.imshow(edges)
        plt.show()
        if type=="dirty":
            plt.imshow(edges2)
            plt.show()

    file_name = f'images/canny/{type}/{id}.jpg'
    cv.imwrite(file_name, edges)

    return file_name

def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    storage_client = storage.Client()
    bucket_name = "train_test_dataset"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def get_coords(id, location, n, radius):
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
        coordinates.append((lat, lon, zoom, id))

    # print(coordinates)
    return coordinates

if __name__ == "__main__":
    locations = ["Northeastern University"]
    coords = []
    id = 0
    for l in locations:
        coords.extend(get_coords(id, l, n=20, radius=1))
        id = id + 1

    temp_limit = 20
    count = 1
    for id, c in enumerate(coords):
        center = str(c[0]) + "," + str(c[1])
        zoom = c[2]
        #id = c[3]

        """ Clean Map """
        clean_image = get_map(center, zoom, "clean", display=False)
        canny_clean_image = apply_canny(clean_image, "clean", id, display=False)
        upload_blob(canny_clean_image, f"clean_train/clean_{id}.jpg")

        """ Dirty Map """
        dirty_image = get_map(center, zoom, "dirty", display=False)
        canny_dirty_image = apply_canny(dirty_image, "dirty", id, display=False)
        upload_blob(canny_dirty_image, f"satellite_train/dirty_{id}.jpg")

        count = count + 1
        if (count > temp_limit):
            break