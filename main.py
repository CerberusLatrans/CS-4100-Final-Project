from google.cloud import aiplatform
import numpy as np
from PIL import Image
from model import ConvAE
import torch
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
from dataset import get_api_key, get_map
import tkinter as tk
from tkinter import simpledialog
from easygui import *
 
PROJECT_NUMBER = "careful-striker-367620"
ENDPOINT_ID = None

#sends the image to the trained model endpoint and returns the output Image
def denoise_cloud(input):
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}")
    x_test = np.asarray(input).astype(np.float32)
    output = endpoint.predict(instances=x_test).predictions
    output_image = Image.fromarray(output)

    return output_image

def denoise(input, weight_path):
    x_test = np.asarray(input).astype(np.float32)
    model = ConvAE(512)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    output = model(x_test)
    output_image = Image.fromarray(output)

    return output_image

# allow user to click starting and ending point on map
def get_coordinates_from_user_clicks(map_image):
    coords = []
    fig = plt.figure(figsize=(10,8))

    def on_click(event):
        global ix, iy

        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            print ('x = %d, y = %d'%(ix, iy))
            plt.plot(event.xdata, event.ydata, 'r*')
            fig.canvas.draw_idle()
            coords.append((ix, iy))

        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(0.1)
            plt.close()

    plt.imshow(map_image.rotate(180).transpose(method=Image.FLIP_LEFT_RIGHT), origin="lower")
    plt.title("Click starting and ending point on map!")
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if len(coords) < 2:
        return None
    return coords

# based on user's input of location and zoom, get the map (with labels)
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
                                    prompt="Enter a location:", initialvalue="Northeastern University") or location_default
    zoom = simpledialog.askinteger("Zoom", "Enter zoom level (between 17 and 20, inclusive)", initialvalue=17) or zoom_default

    map = get_map_from_location(location, zoom, False)
    coords = get_coordinates_from_user_clicks(map)

    if (coords is None):
        print("Must pick a starting and ending point!")
    else:
        start = coords[0]
        end = coords[1]
        print("Starting: " + str(start))
        print("Ending: " + str(end))

        # Do rest of the process
        #clean_map = get_map(location, zoom, 'clean', True)