from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import requests
from dataset import get_api_key
import tkinter as tk
from tkinter import simpledialog
from easygui import *
 

def get_coordinates_from_user_clicks(map_image):
    coords = []
    fig = plt.figure(figsize=(10,8))

    def on_click(event):
        global ix, iy

        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            print ('x = %d, y = %d'%(ix, iy))
            coords.append((ix, iy))

        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
        return coords

    cid = plt.connect('button_press_event', on_click)
    plt.imshow(map_image.rotate(180).transpose(method=Image.FLIP_LEFT_RIGHT), origin="lower")
    plt.title("Click starting and ending point on map!")
    plt.show()

    if len(coords) < 2:
        return None
    return coords

def get_map_from_location(location, zoom, display=False):
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    size = "640x640"
    map_type = "hybrid"
    api_key = get_api_key()

    full_url = url + "center=" + location + "&zoom=" + str(zoom) + "&size=" + size + "&key=" + api_key + "&maptype=" + map_type
    
    r = requests.get(full_url)
    img = Image.open(BytesIO(r.content))

    if display:
        plt.imshow(img)
        plt.show()

    return img


if __name__ == "__main__":
    ROOT = tk.Tk()
    ROOT.withdraw()

    location_default = 'Northeastern University'
    zoom_default = 17

    # the input dialogs
    location = simpledialog.askstring(title="Location",
                                    prompt="Enter a location:") or location_default
    zoom = simpledialog.askinteger("Zoom", "Enter zoom level (between 17 and 20, inclusive)") or zoom_default

    map = get_map_from_location(location, zoom, False)
    coords = get_coordinates_from_user_clicks(map)

    if (coords is None):
        print("Must pick a starting and ending point!")
    else:
        start = coords[0]
        end = coords[1]
        print("Starting: " + str(start))
        print("Ending: " + str(end))