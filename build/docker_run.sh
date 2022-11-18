#!/bin/bash
# Run project dev container
#  ARG 1: <PATH TO LOCAL PROJECT FOLDER>
#  ARG 2: <PATH TO LOCAL PROJECT DATA>

docker run --gpus=all \
           --privileged \
           --shm-size=1g \
           --ulimit memlock=-1 \
           --network=host \
           --rm \
           -it \
           -v $1:/app/project/ \
           -v $2:/app/data \
           sdc-project-1-dev \
           /bin/bash