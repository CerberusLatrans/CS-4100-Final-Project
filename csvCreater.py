import pandas as pd
from google.cloud import storage
from dataset import upload_blob


"""Lists all the blobs in the bucket."""
# bucket_name = "your-bucket-name"

storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
blobs = storage_client.list_blobs('train_test_dataset')

data = {
    {
        'dirty': 
            blobs, 
        'clean': 
            blobs
    }
}
df = pd.DataFrame(data)
df['dirty'] = df['dirty'].str.contains('.')
df['dirty'] = df['dirty'].str.contains('dirty')
df['clean'] = df['clean'].str.contains('.')
df['clean'] = df['clean'].str.contains('clean')



df = pd.DataFrame(data)
csv_data = df.to_csv('dataset.csv')
upload_blob(csv_data, 'dataset.csv')

