#!/bin/bash

# Gradnja Docker slike
sudo docker build . -t $DOCKER_USERNAME/$REPONAME:latest

# Potiskanje slike v DockerHub
docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD docker.io

echo "docker build . --file Dockerfile --tag $DOCKER_USERNAME/$REPO_NAME:latest"
docker build . --file Dockerfile --tag $DOCKER_USERNAME/$REPO_NAME:latest
docker push $DOCKER_USERNAME/$REPO_NAME:latest

