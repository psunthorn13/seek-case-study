from vector_database.pinecone_handler import PineconeHandler

if __name__ == '__main__':
    handler = PineconeHandler(
        data_path='data/ads-50k.json',
        index_name='seek-ads'
    )
    # handler.create_index(field_map={"text": "embed_text"})
    handler.upsert_records(namespace="job-description-namespace")