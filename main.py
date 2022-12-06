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
from dataset import apply_canny, get_api_key, get_map
from search import run_search, euclideanHeuristic, manhattanHeuristic
import tkinter as tk
from tkinter import simpledialog
from easygui import *
import os
import cv2
 
PROJECT_NUMBER = "careful-striker-367620"
ENDPOINT_ID = None

PARAM_PATH = "model-output-one-hour"

#sends the image to the trained model endpoint and returns the output Image
def denoise_cloud(input):
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}")
    x_test = [np.asarray(input).astype(np.float32)]
    output = endpoint.predict(instances=x_test).predictions
    output_image = Image.fromarray(output)

    return output_image

def denoise(input, weight_path , show=False):
    x_test = np.asarray([[input]]).astype(np.float32)
    model = ConvAE(512)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    output = model(torch.Tensor((x_test)))
    output = np.squeeze(output.detach().numpy())
    output_image = Image.fromarray(output)
    if show:
        plt.imshow(output_image)
        plt.show()

    return output

# allow user to click starting and ending point on map
def get_coordinates_from_user_clicks(map_image):
    coords = []
    fig = plt.figure(figsize=(10,8))

    def on_click(event):
        global ix, iy

        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            print ('x = %d, y = %d'%(ix, iy))
            plt.plot(event.xdata, event.ydata, 'r*', markersize=10)
            fig.canvas.draw_idle()
            coords.append((ix, iy))

        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.pause(0.1)
            plt.close()

    plt.imshow(map_image)#map_image.rotate(180).transpose(method=Image.FLIP_LEFT_RIGHT), origin="lower")
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
    zoom = simpledialog.askinteger(title="Zoom", prompt="Enter zoom level (between 17 and 20, inclusive)", initialvalue=17) or zoom_default
    ROOT.destroy()

    map = get_map_from_location(location, zoom, False)
    coords = get_coordinates_from_user_clicks(map)

    if (coords is None):
        print("Must pick a starting and ending point!")
    else:
        start = coords[0]
        end = coords[1]

        start = [start[0]/640, start[1]/640]
        end = [end[0]/640, end[1]/640]

        print("Starting: " + str(start))
        print("Ending: " + str(end))

        """ Do rest of the process """
        # Get and show dirty map without labels
        dirty_img_arr, dirty_img = get_map(location, zoom, 'dirty', display=True)

        # Get and show canny applied to dirty map
        dirty_edges_path = apply_canny(dirty_img_arr, "dirty", "userInput", display=True)
        dirty_edges = np.asarray(Image.open(dirty_edges_path))
        os.remove(dirty_edges_path)

        # Get and show denoising autoencoder applied
        clean_edges = denoise(dirty_edges, PARAM_PATH)

        # Run A* and show A* path on original image
        res = 250
        start = [int(start[0]*res), int(start[1]*res)]
        end = [int(end[0]*res), int(end[1]*res)]

        path, found = run_search(clean_edges, resolution=res, start=start, end=end,heuristic=euclideanHeuristic, display=True)

        # Scale x and y coordinates in path for 640x640 area
        ratio = 650 / res 
        x_coords = []
        y_coords = []
        for coord in path:
            x_coords.append(coord[1] * ratio)
            y_coords.append(coord[0] * ratio)

        fig = plt.figure(figsize=(10,8))
        plt.imshow(dirty_img)
        plt.plot(x_coords, y_coords, color='blue', marker='.', linestyle='solid', linewidth=3, markersize=3) # Plot points connected by line
        if found:
            title_str = 'A path was found!'
        else:
            title_str = 'Sorry, no path was found :('
        plt.title(title_str)
        plt.show()