from google.cloud import aiplatform
import numpy as np
from PIL import Image
from model import ConvAE
import torch

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