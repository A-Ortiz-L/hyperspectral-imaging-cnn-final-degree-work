from google.cloud import bigquery
from config.cfg import table, data_set


class GoogleBigQuery:

    def __init__(self):
        self.client = bigquery.Client()
        self.table_ref = self.client.dataset(data_set).table(table)
        self.table = self.client.get_table(self.table_ref)

    def insert_row(self, row):
        self.client.insert_rows(self.table, row)
