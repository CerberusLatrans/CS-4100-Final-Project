import pandas as pd
from google.cloud import storage
from dataset import upload_blob


"""Lists all the blobs in the bucket."""
# bucket_name = "your-bucket-name"

storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
blobs = storage_client.list_blobs('train_test_dataset')
# blob_names = blobs.name
list = [element.name for element in blobs]

data = {
    'dirty': 
        list, 
    'clean': 
        list
    
}
df = pd.DataFrame(data)
df['dirty'] = df[df['dirty'].str.contains('.') == True]
df['dirty'] = df[df['dirty'].str.contains('dirty') == True]
df['clean'] = df[df['clean'].str.contains('.') == True]
df['clean'] = df[df['clean'].str.contains('clean') == True]



# df = pd.DataFrame(data)
csv_data = df.to_csv('dataset.csv')
upload_blob('dataset.csv', 'dataset.csv')

