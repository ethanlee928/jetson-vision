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
    ticker = FPSTicker()

    start = sv.Point(1280 // 2, 0)
    end = sv.Point(1280 // 2, 720)
    line_zone = sv.LineZone(start=start, end=end)
    line_zone_annotator = sv.LineZoneAnnotator()

    tracker = sv.ByteTrack()

    while True:
        img = input.Capture()
        ticker.tick()
        if img is None:
            continue

        detections = net.Detect(img, overlay=opt.overlay)
        img_np = jetson_utils.cudaToNumpy(img)
        sv_detections = to_sv_detections(detections)
        if sv_detections:
            sv_detections = tracker.update_with_detections(sv_detections)
            line_zone.trigger(sv_detections)
            img_np = line_zone_annotator.annotate(img_np, line_counter=line_zone)
            jetson_utils.Log.Info(
                f"In: {line_zone.in_count}, Out: {line_zone.out_count}"
            )
        img_cuda = jetson_utils.cudaFromNumpy(img_np)
        output.Render(img_cuda)

        if not input.IsStreaming() or not output.IsStreaming():
            break
