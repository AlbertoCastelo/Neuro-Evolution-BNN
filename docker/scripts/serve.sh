#!/usr/bin/env bash

# remove previous container if exists
docker rm -f pyneat

# build
# docker build . -t jupy-lab --no-cache
docker build ./docker/ -t pyneat

# serve interactively
docker run -it -p 8888:8888 --name pyneat -v "$PWD":/home/jovyan pyneat