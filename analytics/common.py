import argparse
import sys
from datetime import datetime
from typing import List, Optional

import supervision as sv
import numpy as np
import jetson_inference
import jetson_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate objects in a live camera stream using an object detection DNN."
    )
    parser.add_argument(
        "input_URI", type=str, default="", nargs="?", help="URI of the input stream"
    )
    parser.add_argument(
        "output_URI", type=str, default="", nargs="?", help="URI of the output stream"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="ssd-mobilenet-v2",
        help="pre-trained model to load (see below for options)",
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default="box,labels,conf",
        help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum detection threshold to use",
    )

    try:
        opt = parser.parse_known_args()[0]
        return opt
    except:
        print("")
        parser.print_help()
        sys.exit(0)


def to_sv_detections(
    detections: List[jetson_inference.detectNet.Detection],
) -> Optional[sv.Detections]:
    if len(detections) == 0:
        return None
    confidences, class_ids, tracker_ids, bboxes = [], [], [], []
    for detection in detections:
        confidences.append(detection.Confidence)
        class_ids.append(detection.ClassID)
        tracker_ids.append(detection.TrackID)
        bboxes.append(
            [detection.Left, detection.Top, detection.Right, detection.Bottom]
        )
    return sv.Detections(
        xyxy=np.array(bboxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences),
        tracker_id=np.array(tracker_ids),
    )


class FPSTicker:
    def __init__(self, interval: int = 2) -> None:
        self._interval = interval
        self._last_tick = datetime.now()
        self._count = 0
        self._fps = 0

    def tick(self) -> None:
        self._count += 1
        now = datetime.now()
        dt = now - self._last_tick
        if dt.total_seconds() > self._interval:
            self._fps = self._count / dt.total_seconds()
            jetson_utils.Log.Success(f"FPS: {self._fps:.2f}")
            self._last_tick = now
            self._count = 0
