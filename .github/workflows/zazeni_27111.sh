#!/bin/bash

# Pridobivanje slike iz DockerHub
sudo docker pull $DOCKER_USERNAME/$REPONAME:latest

# Zagon kontejnerja
sudo docker run -ti --name test -v `pwd`/share:/mnt/share $DOCKER_USERNAME/$REPONAME:latest
