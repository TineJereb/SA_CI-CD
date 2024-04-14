#!/bin/bash

# Gradnja Docker slike
sudo docker build . -t $DOCKER_USERNAME/$REPONAME:latest

# Potiskanje slike v DockerHub
sudo docker login -u $DOCKER_USERNAME -p $DOCKER_PASSWORD docker.io
sudo docker push $DOCKER_USERNAME/$REPONAME:latest
