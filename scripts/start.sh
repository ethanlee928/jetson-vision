#!/bin/bash

modelDir="${modelDir:-${PWD}/networks}"

docker run -it --rm --network host -v ${PWD}:/app/ \
    -v ${modelDir}:/jetson-inference/data/networks \
    -v /run/jtop.sock:/run/jtop.sock \
    -v /dev/video0:/dev/video0 \
    --device /dev/video0 \
    -w /app/ \
    jetson-vision \
    bash
