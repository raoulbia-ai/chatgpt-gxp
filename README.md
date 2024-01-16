21 CFR Part 11 (up to date as of 1-09-2024).pdf

# Build the Docker image
docker build -t my-python-app .

# Test the Docker container locally
docker run -p 8501:8501 my-python-app

# Push the Docker image to Docker Hub
docker tag my-python-app:latest mydockerhubusername/my-python-app:latest
docker push mydockerhubusername/my-python-app:latest

# Or push the Docker image to Azure Container Registry
# docker tag my-python-app:latest myacrname.azurecr.io/my-python-app:latest
# docker push myacrname.azurecr.io/my-python-app:latest

# Create an Azure Container Instance
# Replace 'myresourcegroup' with your Azure Resource Group name
# Replace 'mycontainername' with your desired container name
# Replace 'mydockerhubusername/my-python-app:latest' with your image name
az container create --resource-group myresourcegroup --name mycontainername --image mydockerhubusername/my-python-app:latest --dns-name-label mycontainername --ports 8501