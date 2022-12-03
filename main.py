from google.cloud import aiplatform
import numpy as np
from PIL import Image

PROJECT_NUMBER = "careful-striker-367620"
ENDPOINT_ID = None

#sends the image to the trained model endpoint and returns the output Image
def denoise(input):
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}")
    x_test = np.asarray(input).astype(np.float32).tolist()
    output = endpoint.predict(instances=x_test).predictions
    output_image = Image.fromarray(output)

    return output_image