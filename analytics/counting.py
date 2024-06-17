import sys

import numpy as np
import supervision as sv
import jetson_inference
import jetson_utils

from common import parse_args, to_sv_detections, FPSTicker


def get_roi() -> np.ndarray:
    return np.array([0, 720 // 2, 1280 // 2, 720 // 2, 1280 // 2, 720, 0, 720]).reshape(-1, 2)


if __name__ == "__main__":
    opt = parse_args()

    net = jetson_inference.detectNet(opt.network, sys.argv, opt.threshold)

    input = jetson_utils.videoSource(opt.input_URI, argv=sys.argv)
    output = jetson_utils.videoOutput(opt.output_URI, argv=sys.argv)
    ticker = FPSTicker()

    polygon_zone = sv.PolygonZone(polygon=get_roi())
    polygon_zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.ROBOFLOW)
    round_box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    tracker = sv.ByteTrack()

    while True:
        img = input.Capture()
        ticker.tick()
        if img is None:
            continue

        detections = net.Detect(img, overlay="none")
        img_np = jetson_utils.cudaToNumpy(img)
        sv_detections = to_sv_detections(detections)
        if sv_detections:
            sv_detections = tracker.update_with_detections(sv_detections)
            ppl_detections = sv_detections[sv_detections.class_id == 1]
            mask = polygon_zone.trigger(ppl_detections)
            ppl_in_zone_detections = ppl_detections[mask]
            img_np = polygon_zone_annotator.annotate(img_np)
            img_np = round_box_annotator.annotate(img_np, detections=ppl_in_zone_detections)
            img_np = label_annotator.annotate(img_np, detections=ppl_in_zone_detections)

        img_cuda = jetson_utils.cudaFromNumpy(img_np)
        output.Render(img_cuda)

        if not input.IsStreaming() or not output.IsStreaming():
            break
