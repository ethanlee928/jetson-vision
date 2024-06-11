from jetson_utils import videoOutput, videoSource

if __name__ == "__main__":
    video_source = videoSource("/dev/video0")
    video_output = videoOutput("rtsp://@:8554/camera")

    while True:
        image = video_source.Capture(format="rgb8", timeout=1000)
        if image is None:
            continue

        video_output.Render(image)
        if not video_source.IsStreaming() or not video_output.IsStreaming():
            break
