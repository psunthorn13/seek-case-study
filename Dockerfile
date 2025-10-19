# Start from the official Python image.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Command to run the app
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
