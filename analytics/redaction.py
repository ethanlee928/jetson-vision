import sys

import supervision as sv
import jetson_inference
import jetson_utils

from common import parse_args, to_sv_detections, FPSTicker


if __name__ == "__main__":
    opt = parse_args()
    net = jetson_inference.detectNet(opt.network, sys.argv, opt.threshold)
    input = jetson_utils.videoSource(opt.input_URI, argv=sys.argv)
    output = jetson_utils.videoOutput(opt.output_URI, argv=sys.argv)
    pixelate_annotator = sv.PixelateAnnotator()
    ticker = FPSTicker()

    while True:
        img = input.Capture()
        ticker.tick()
        if img is None:
            continue

        detections = net.Detect(img, overlay="none")
        img_np = jetson_utils.cudaToNumpy(img)
        sv_detections = to_sv_detections(detections)
        if sv_detections:
            person_detections = sv_detections[sv_detections.class_id == 1]
            if person_detections:
                img_np = pixelate_annotator.annotate(
                    scene=img_np, detections=person_detections
                )

        img_cuda = jetson_utils.cudaFromNumpy(img_np)

        output.Render(img_cuda)

        if not input.IsStreaming() or not output.IsStreaming():
            break
