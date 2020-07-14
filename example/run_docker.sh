#!/bin/sh
docker run --rm \
          -it \
          -p 8888:8888 \
          -e JUPYTER_ENABLE_LAB=yes \
          --name injectTF_dev \
          -v "$(pwd)/..:/home/jovyan" \
          nvaitc/ai-lab:19.10-tf2 \
          bash
