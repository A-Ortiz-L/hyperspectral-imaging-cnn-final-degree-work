from google.cloud import storage
from google.cloud.exceptions import NotFound

from logging import getLogger

log = getLogger(__name__)


class GoogleStorage:
    def __init__(self):
        self.client = storage.Client()
        self.storage_list = {}

    def get_bucket(self, bucket_name: str):
        try:
            bucket = self.client.get_bucket(bucket_name)
            return bucket
        except NotFound:
            log.warning(f'Could not find bucket={bucket_name}')
            return False

    def download_blob(self, bucket_name: str, source_blob_name: str, destination_file_name: str) -> bool:
        """Downloads a blob from the bucket."""
        bucket = self.get_bucket(bucket_name)
        if not bucket or not bucket.get_blob(source_blob_name):
            log.warning(f'Could not download blob={source_blob_name} on bucket={bucket_name}')
            return False
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        log.info('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))
        return True
