#!/bin/bash

# Pridobivanje slike iz DockerHub
sudo docker pull $DOCKER_USERNAME/$REPO_NAME:latest

# Zagon kontejnerja
sudo docker run --name test -v `pwd`/share:/mnt/share $DOCKER_USERNAME/$REPO_NAME:latest
