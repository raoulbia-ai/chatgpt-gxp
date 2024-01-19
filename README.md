#### Docker commands

Name of container registry: `gptserve`

Name of Repository: `chatgpt_gxp`

Name of WebApp: `gptgxp`

Name of resource group: `gptserve-rg`

Login command: `az acr login --name gptserve` or `docker login gptserve.azurecr.io`

```
docker build -t gptserve.azurecr.io/chatgpt_gxp:v2 .

docker push gptserve.azurecr.io/chatgpt_gxp:v2
```
You can now either deploy the Docker container using the code below, or go to the Portal and manually update the tag 
version in the Deployment Center. 
```
az functionapp config container set 
--name gptgxp 
--resource-group gptserve-rg 
--docker-custom-image-name gptserve.azurecr.io/chatgpt_gxp:v2
--docker-registry-server-url https://gptserve.azurecr.io --docker-registry-server-user gptserve --docker-registry-server-password <PASSWORD>
```

Note: to get the password for the ACR login above use `az acr credential show --name` 

To run the Docker image locally: `docker run -p 8080:80 gptserve.azurecr.io/chatgpt_gxp:v2`