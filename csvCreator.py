import pandas as pd
from google.cloud import storage
from dataset import upload_blob

storage_client = storage.Client()

"""Lists all the blobs in the bucket."""
blobs = storage_client.list_blobs('train_test_dataset')

clean_list = []
dirty_list = []

for element in blobs:
    blob_name = element.name
    if '.' in blob_name and 'clean' in blob_name:
        clean_list.append(blob_name)
    elif '.' in blob_name and 'dirty' in blob_name:
        dirty_list.append(blob_name)

data = {
    'dirty': 
        dirty_list, 
    'clean': 
        clean_list
}


df = pd.DataFrame(data) # Dataframe containing dirty and clean filenames according to GCS structure
csv_data = df.to_csv('dataset.csv')
upload_blob('dataset.csv', 'dataset.csv') # Uploads CSV to GCS

