
# Document Q&A PoC

This repository contains a Streamlit application that uses a question-answering model to provide responses based on a set of regulatory documents. The application is  
designed to handle questions related to GxP regulatory guidelines and pharmaceutical compliance. The model's knowledge is based on specific documents including FDA     
Title 21 CFR Part 11, FDA's GAMP 5 Guide, EU's Annex 11, EMA's Cloud Strategy, and EMA's Guideline on Quality Risk Management (Q9). 

## Deployment

### Docker Commands for `gptserve` Container Registry

This document provides instructions on how to log into the `gptserve` Azure Container Registry (ACR) and interact with the `chatgpt_gxp` repository.

### Resources

- Container Registry: `gptserve`
- Repository: `chatgpt_gxp`
- WebApp: `gptgxp`
- Resource Group: `gptserve-rg`

### Logging into the Registry

You can log into the ACR using either the Azure CLI or Docker. Here are the commands for both:

- Azure CLI: `az acr login --name gptserve`
- Docker: `docker login gptserve.azurecr.io`

The username and password for logging in are associated with the `gptserve` ACR.

### Retrieving Credentials

#### Azure Portal

1. Navigate to your Azure Container Registry resource.
2. Click on "Access keys" in the left-hand menu.
3. Here you'll find the "Login server", "Username", and two passwords (password and password2). You can use either of the two passwords.

#### Azure CLI

If you have the Azure CLI installed, you can retrieve the credentials using the following command:

`az acr credential show --name gptserve`