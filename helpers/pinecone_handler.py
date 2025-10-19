from pinecone import Pinecone

import pandas as pd
from helpers.data_loader import DataLoader
import os
import time


class PineconeHandler:
    """Handler for Pinecone operations including index creation, data upsertion, and searching."""
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

    def __init__(self,
                 index_name: str,
                 data_path: str | None = None,
                 cloud: str = "aws",
                 region: str = "us-east-1",
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

    def create_index(self, field_map: dict):
        """Create a Pinecone index with the specified field mapping for embeddings."""
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

    def upsert_records(self, namespace: str, batch_size: int = 20):
        """Upsert records into the Pinecone index in batches."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)

        if not self.records:
            print("No records to upsert")
            return
        self.records = self.records
        total_records = len(self.records)
        total_batches = (total_records - 1) // batch_size + 1
        print(f"Upserting {total_records} records in {total_batches} batches")

        # Upsert in batches to avoid rate limiting
        for i in range(0, total_records, batch_size):
            batch = self.records[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} with {len(batch)} records")
            time.sleep(1)  # To avoid rate limiting

            try:
                self.index.upsert_records(namespace, batch)
                print(f"Successfully upserted batch {batch_num}")
            except Exception as e:
                print(f"Error upserting batch {batch_num}: {str(e)}")
                raise

    @staticmethod
    def convert_search_results_to_dataframe(search_results) -> pd.DataFrame:
        """Convert Pinecone search results to a pandas DataFrame."""
        # Extract all records
        records = []

        # Iterate through search result
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

    def search(self,
               namespace: str,
               query: str,
               top_k: int = 10,
               filter_dict: dict = None) -> pd.DataFrame:
        """Search the Pinecone index with the given query and return results as a DataFrame."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        search_results = self.index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    'text': query
                },
                "filter": filter_dict
            }
        )

        search_df = self.convert_search_results_to_dataframe(search_results)
        return search_df


if __name__ == '__main__':
    handler = PineconeHandler(
        data_path='data/ads-50k.json',
        index_name='seek-ads'
    )

    results = handler.search(
        namespace="job-description-namespace",
        query="Data Scientist jobs in Melbourne that uses Python and machine learning"
    )
    print(results)
