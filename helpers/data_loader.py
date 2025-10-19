import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


class DataLoader:
    def __init__(self, file_path: str):
        """Initialise with the path to the JSON data file."""
        self.file_path = Path(file_path)

    def _load_json_to_df(self) -> pd.DataFrame:
        """Read the JSON file and convert it to a pandas DataFrame with flattened metadata."""
        # Read the JSON file
        with open(self.file_path, 'r') as f:
            records = [json.loads(line) for line in f]

        # Convert to DataFrame
        job_df = pd.DataFrame(records)

        # Flatten the metadata column if it exists
        if 'metadata' in job_df.columns:
            metadata_df = pd.json_normalize(job_df['metadata']).add_prefix('metadata.')
            job_df = pd.concat([job_df, metadata_df], axis=1)

        return job_df

    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags from text data."""
        if pd.isna(text):
            return text

        # Remove HTML tags
        clean_text = re.sub(r'<.*?>', ' ', str(text))
        # Replace multiple spaces with a single space
        clean_text = re.sub(r'\s+', ' ', clean_text)
        # Strip leading and trailing spaces
        clean_text = clean_text.strip()

        return clean_text

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning operations to the DataFrame."""
        # Replace empty strings with NaN
        df = df.replace('', pd.NA)
        if 'content' in df.columns:
            df['cleaned_content'] = df['content'].apply(self.clean_html)
        return df

    def get_processed_data(self) -> pd.DataFrame:
        """Load, process, and return the data."""
        df = self._load_json_to_df()
        return self._process_dataframe(df)

    def _create_embed_text(self) -> pd.DataFrame:
        """Create an embedding text column by combining title and cleaned content."""
        df = self.get_processed_data()

        # Apply formatting row by row
        df['embed_text'] = df.apply(
            lambda row: f"""Job Title: {row['title']}
Classification: {row['metadata.classification.name']}
Location: {row['metadata.location.name']}
Work Type: {row['metadata.workType.name']}
Salary: {row['metadata.additionalSalaryText']}
Description: {row['cleaned_content']}
""",
            axis=1
        )

        return df

    @staticmethod
    def _validate_records(records: list[dict]) -> list[dict]:
        """Validate records to ensure JSON compatibility."""
        validated_records = []

        for i, record in enumerate(records):
            try:
                # Create a clean copy
                clean_record = {}

                for key, value in record.items():
                    # Handle string values that might cause JSON parsing issues
                    if isinstance(value, str):
                        # Replace problematic characters that might break JSON
                        value = value.replace('\x00', '')
                        value = value.replace('\n', ' ').replace('\r', ' ')

                        # Truncate extremely long string values to avoid issues
                        if len(value) > 10000:
                            value = value[:10000] + "..."

                    # Skip None values, empty strings, or NaN
                    if value is not None and value != "" and not pd.isna(value):
                        clean_record[key] = value

                # Test if record is valid JSON
                json.dumps(clean_record)
                validated_records.append(clean_record)
            except Exception as e:
                print(f"Invalid record at position {i}: {str(e)}")
                continue

        print(f"Validated {len(validated_records)} out of {len(records)} records")
        return validated_records

    def get_data_for_insertion(self) -> list[dict]:
        """Get data prepared for insertion with embedding text."""
        df = self._create_embed_text()
        df = df.drop(columns=['cleaned_content', 'metadata'], errors='ignore')
        df = df.rename(columns={'id': '_id'})
        records = df.to_dict(orient='records')
        validated_records = self._validate_records(records)
        return validated_records


@st.cache_data(show_spinner=False)
def load_data(file_path='data/ads-50k.json'):
    """Load and process data with caching for Streamlit."""
    loader = DataLoader(file_path)
    return loader.get_processed_data()
