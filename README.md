# Jetson Vision

Real time video analytics with Nvidia's Jetson devices.

## Prerequisites

### Hardware

- Tested with NVIDIA Jetson Nano (Jetpack 4.6 [L4T 32.6.1])

### Docker Nvidia Runtime

```bash
sudo vim /etc/docker/daemon.json

{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

```bash
sudo service docker restart

# Check
sudo docker info | grep Default

# Expected output
Default Runtime: nvidia
WARNING: No blkio weight support
WARNING: No blkio weight_device support
```

## How to Start

### Docker Enviornment

```bash
./scripts/build.sh

# Start docker container
./scripts/start.sh
```

### Download Pre-trained Models

- Pretrained models will be downloaded @ `/jetson-inference/data/networks`
- In `scripts/start.sh`, the models directory is mounted to local volume (`/media/data/models/`). Thus, no need to re-download the models multiple times in docker environment.

```bash
cd /jetson-inference/tools
./download-models.sh
```

## Basics

Hello world codes for using Jetson-inference.

```bash
cd basics
```

### Object Detection

```bash
python3 detect.py /dev/vidoe0
```

### Semantic Segmentation

```bash
python3 segment.py /dev/video0
```

## Analytics

Using Jetson-inference toghether with Supervision to do vidoe analytics.

```bash
cd analytics/
```

### People Counting in a Zone

Counting number of people in a defined polygon zone.

```bash
python3 counting.py /dev/video0
```

### Flow analysis

Counting objects going in and going out of a line zone.

```bash
python3 flow.py /dev/video0
```

### People Redaction

Detects person and redact the whole body, could be used to process video with privacy concerns.

```bash
python3 redaction.py /dev/video0
```
