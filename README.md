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

## Start Environment

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

### Object Detection

```bash
python3 detect.py /dev/vidoe0
```

### Semantic Segmentation

```bash
python3 segment.py /dev/video0
```
