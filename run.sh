#!/bin/bash
echo "Running docker image for serving and streamlit"
export $(cat .env | xargs)
docker run -e COMET_API_KEY -e COMET_ML_PROJECT_NAME -e COMET_ML_WORKSPACE -p 8000:8000 --net=host serving & docker run -p 8080:8080 --net=host streamlit
# echo "TODO: fill in the docker run command"