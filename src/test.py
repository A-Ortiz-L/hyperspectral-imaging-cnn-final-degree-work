from google.cloud import bigquery
import os
# Construct a BigQuery client object.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './config/key.json'
client = bigquery.Client()

# TODO(developer): Set dataset_id to the ID of the dataset to create.
dataset_id = "{}.your_dataset".format(client.project)

# Construct a full Dataset object to send to the API.
dataset = bigquery.Dataset(dataset_id)

# TODO(developer): Specify the geographic location where the dataset should reside.
dataset.location = "EU"

# Send the dataset to the API for creation.
# Raises google.api_core.exceptions.Conflict if the Dataset already
# exists within the project.
dataset = client.create_dataset(dataset)  # Make an API request.
print("Created dataset {}.{}".format(client.project, dataset.dataset_id))