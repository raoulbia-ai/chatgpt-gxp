# Use the appropriate base image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the application files and the .env file into the container
COPY . /app
COPY .env /app/.env


# Copy the storageDefaultLlmAll directory into the container
COPY storageDefaultLlmAll /app/storageDefaultLlmAllJSON

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# LOCAL DEV ONLY!
ENV SSL_CERT_FILE=./ZscalerRootCertificate-2048-SHA256.pem 

# Start the Streamlit application
CMD ["streamlit", "run", "app_json.py", "--server.port=80"]
