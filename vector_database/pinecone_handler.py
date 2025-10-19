from pinecone import Pinecone

import pandas as pd
from helpers.data_loader import DataLoader
import os
import logging
import time



class PineconeHandler:
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

    def __init__(self, index_name: str, data_path: str | None = None, cloud: str = "aws", region: str = "us-east-1",
                 model: str = "llama-text-embed-v2"):

        self.data_path = data_path
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.model = model

        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        if data_path:
            self.data_loader = DataLoader(self.data_path)
            self.records = self.data_loader.get_data_for_insertion()
        self.index = None

    def create_index(self, field_map):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud=self.cloud,
                region=self.region,
                embed={
                    "model": self.model,
                    "field_map": field_map
                }
            )
        self.index = self.pc.Index(self.index_name)

    def upsert_records(self, namespace, batch_size=20):
        logger = logging.getLogger(__name__)

        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        if not self.records:
            logger.warning("No records to upsert")
            return
        self.records = self.records[10000:]
        total_records = len(self.records)
        total_batches = (total_records - 1) // batch_size + 1
        logger.info(f"Upserting {total_records} records in {total_batches} batches")

        for i in range(0, total_records, batch_size):
            batch = self.records[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} with {len(batch)} records")
            time.sleep(1)  # To avoid rate limiting

            try:
                self.index.upsert_records(namespace, batch)
                logger.info(f"Successfully upserted batch {batch_num}")
            except Exception as e:
                logger.error(f"Error upserting batch {batch_num}: {str(e)}")
                raise

    @staticmethod
    def convert_search_results_to_dataframe(search_results):
        """Convert Pinecone search results to a pandas DataFrame."""
        # Extract all records
        records = []

        for item in search_results.result.hits:
            # Start with the _id and _score
            record = {
                'id': item['_id'],
                '_score': item['_score']
            }

            # Add all fields from the nested fields dictionary
            if 'fields' in item:
                for key, value in item['fields'].items():
                    record[key] = value

            records.append(record)

        # Create DataFrame
        return pd.DataFrame(records)
    def search(self, namespace, query, top_k=10):
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        search_results = self.index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    'text': query
                }
            }
        )

        search_df = self.convert_search_results_to_dataframe(search_results)
        return search_df


if __name__ == '__main__':
    handler = PineconeHandler(
        data_path='data/ads-50k.json',
        index_name='seek-ads'
    )
    # handler.create_index(field_map={"text": "embed_text"})
    # handler.upsert_records(namespace="job-description-namespace")
    results = handler.search(
        namespace="job-description-namespace",
        query="Data Scientist jobs in Melbourne that uses Python and machine learning"
    )
    print(results)
