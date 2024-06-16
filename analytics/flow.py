import argparse
import sys
from typing import List, Optional

import supervision as sv
import numpy as np
import jetson_inference
import jetson_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN."
    )
    parser.add_argument("input_URI", type=str, default="", nargs="?", help="URI of the input stream")
    parser.add_argument("output_URI", type=str, default="", nargs="?", help="URI of the output stream")
    parser.add_argument(
        "--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)"
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default="box,labels,conf",
        help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

    try:
        opt = parser.parse_known_args()[0]
        return opt
    except:
        print("")
        parser.print_help()
        sys.exit(0)

def to_sv_detections(detections: List[jetson_inference.detectNet.Detection]) -> Optional[sv.Detections]:
    if len(detections) == 0:
        return None
    confidences, class_ids, tracker_ids, bboxes = [], [], [], []
    for detection in detections:
        confidences.append(detection.Confidence)
        class_ids.append(detection.ClassID)
        tracker_ids.append(detection.TrackID)
        bboxes.append([detection.Left, detection.Top, detection.Right, detection.Bottom])
    return sv.Detections(xyxy=np.array(bboxes), class_id=np.array(class_ids), confidence=np.array(confidences), tracker_id=np.array(tracker_ids))


if __name__ == "__main__":
    opt = parse_args()

    # load the object detection network
    net = jetson_inference.detectNet(opt.network, sys.argv, opt.threshold)

    # create video sources & outputs
    input = jetson_utils.videoSource(opt.input_URI, argv=sys.argv)
    output = jetson_utils.videoOutput(opt.output_URI, argv=sys.argv)

    # SV linezone
    start = sv.Point(1280 // 2, 0)
    end = sv.Point(1280 // 2, 720)
    line_zone = sv.LineZone(start=start, end=end)

    # SV tracker
    tracker = sv.ByteTrack()

    # process frames until the user exits
    while True:
        # capture the next image
        img = input.Capture()
        if img is None:
            continue

        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=opt.overlay)
        sv_detections = to_sv_detections(detections)
        if sv_detections:
            sv_detections = tracker.update_with_detections(sv_detections)
            print(sv_detections)
            line_zone.trigger(sv_detections)
            jetson_utils.Log.Success(f"In: {line_zone.in_count}, Out: {line_zone.out_count}")

        # render the image
        output.Render(img)

        # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break
