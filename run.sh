#!/bin/bash
export $(cat .env | xargs)
docker run -e COMET_API_KEY -e COMET_ML_PROJECT_NAME -e COMET_ML_WORKSPACE -p 8000:8000 serving
# echo "TODO: fill in the docker run command"