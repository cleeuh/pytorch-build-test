#!/bin/bash


sudo docker build --platform linux/amd64 -t tmp/cuda_test_116 --load . && time sudo docker run --gpus all -it --platform linux/amd64 tmp/cuda_test_116