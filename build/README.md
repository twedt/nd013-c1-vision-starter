# Instructions

## Requirements

* NVIDIA GPU with the latest driver installed
* docker / nvidia-docker

This build has been tested with Nvidia Drivers 470.141.03 and CUDA 11.4 on a 
Pop!_OS 22.04 LTS machine.

## Build
Build the image with:
```
docker build -t sdc-project-1-dev -f Dockerfile .
```

Create a container with:
```
./docker_run.sh
```