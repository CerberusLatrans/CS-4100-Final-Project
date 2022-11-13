
import requests
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import random
import math
import urllib

API_KEY = "AIzaSyATvdjLWeqJ0svqAXNYmdAmVU7tkaz8434" # Prob want this as a google secret through GCP
URL = "https://maps.googleapis.com/maps/api/staticmap?"
SIZE = "640x640"
MAP_ID = "5fd53f1f4a5c6512" # For styling the map -- clean map
MAP_TYPE = "satellite" # for dirty map
# center = "42.3398106,-71.0913604" <- northeastern university
# labelsTextOff="feature:all|element:labels.text|visibility: off"
# labelsIconOff="feature:all|element:labels.icon|visibility: off"

def get_map(center, zoom, file_name, type):
    full_url = URL + "center=" + center + "&zoom=" + str(zoom) + "&size=" + SIZE + "&key=" + API_KEY
    if (type == 'clean'):
        full_url += "&map_id=" + MAP_ID 
    if (type == 'dirty'):
        full_url += "&maptype=" + MAP_TYPE
    
    r = requests.get(full_url)
    with open('images/pre-canny/' + type + '/' + file_name, 'wb') as file:
        file.write(r.content)
    r.close()

    return file_name
    # using matpltolib to display the image
    # plt.figure(figsize=(5, 5))
    # img=mpimg.imread('map.jpg') #render the terrain
    # dirtyImg=mpimg.imread('dirtyMap.jpg') #render the styled map
    # imgplot = plt.imshow(img)
    # dirtyImgPlot = plt.imshow(dirtyImg)
    # plt.axis('off')
    # plt.show()

def apply_canny(file_name): #TODO
    """Applies canny filter to image"""
    pass

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

locations = ["New York", "Boston"]
coords = []
for l in locations:
    coords.extend(get_coords(l, n=10, radius=5))

temp_limit = 1
count = 1
for c in coords:
    center = str(c[0]) + "," + str(c[1])
    zoom = c[2]

    """ Clean Map """
    clean_image = get_map(center, zoom, "clean_" + center + ".jpg", "clean")  # what's a better name for the image files?
    canny_clean_image = apply_canny(clean_image)
    upload_blob(canny_clean_image)

    """ Dirty Map """
    dirty_image = get_map(center, zoom, "dirty_" + center + ".jpg", "dirty")
    canny_dirty_image = apply_canny(clean_image)
    upload_blob(canny_dirty_image)

    count = count + 1
    if (count > temp_limit):
        break