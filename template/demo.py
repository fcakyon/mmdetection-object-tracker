import argparse

import numpy as np
import torch
from typing import List, Optional

import norfair
from norfair import Detection, Tracker, Video


class Detector:
    def __init__(self, model_path: str, config_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set properties
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        # load model
        self.model = self.load_model()

    def load_model(self):
        NotImplementedError()

    def __call__(self, frame) -> List[Detection]:
        NotImplementedError()


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--max_track_distance", type=str, default="30", help="Maximum tracker distance"
)
parser.add_argument("--model_path", type=str, default="", help="Detector model path")
parser.add_argument("--config_path", type=str, default="", help="Detector model path")
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track_points",
    type=str,
    default="centroid",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

detector = Detector(
    model_path=args.model_path,
    config_path=args.config_path,
    device=args.device,
    track_points=args.track_points,
)

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=args.max_track_distance,
    )

    for frame in video:
        detections = detector(frame)
        tracked_objects = tracker.update(detections=detections)
        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
