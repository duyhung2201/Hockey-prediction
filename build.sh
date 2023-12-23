#!/bin/bash
echo "Building docker image for serving"
docker build -f Dockerfile.serving -t serving .

echo "Building docker image for streamlit"
docker build -f Dockerfile.streamlit -t streamlit .
# echo "TODO: fill in the docker build command"