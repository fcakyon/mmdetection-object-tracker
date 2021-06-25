import argparse
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import norfair
from norfair import Detection, Tracker, Video
from mmdet.apis import init_detector, inference_detector


class Detector:
    def __init__(self, model_path: str, config_path: str, device: Optional[str] = None, track_points: str = "bbox"):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception("Selected device='cuda', but cuda is not available to Pytorch.")
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # set properties
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.track_points = track_points
        # load model
        self.model = self.load_model()

    def load_model(self):
        assert Path(self.model_path).is_file(), TypeError(f"{self.model_path} should be a checkpoint file.")
        assert Path(self.config_path).is_file(), TypeError(f"{self.config_path} should be a config file.")

        if not Path(self.model_path).exists():
            FileNotFoundError(f"{self.model_path} is not present.")

        model = init_detector(config=self.config_path, checkpoint=self.model_path, device=self.device)
        return model

    def _mmdet_result_to_norfair_detections(self, result, conf_thresh: float = 0.35):
        # parse bbox/labels from result
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)

        # init norfair detections
        norfair_detections: List[Detection] = []

        # convert model detections to norfiar detections
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            score = bbox[4]
            if score > conf_thresh:
                bbox = bbox[:4].astype(np.int32)

                if self.track_points == "centroid":
                    centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    scores = np.array([score])
                    norfair_detections.append(Detection(points=centroid, scores=scores))
                elif self.track_points == "bbox":
                    bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                    scores = np.array([score, score])
                    norfair_detections.append(Detection(points=bbox, scores=scores))

        return norfair_detections

    def __call__(self, image, img_scale: int = 720, conf_thresh: float = 0.35) -> List[Detection]:
        self.model.cfg.data.test.pipeline[1]["img_scale"] = (img_scale,)
        result = inference_detector(self.model, image)
        detections = self._mmdet_result_to_norfair_detections(result, conf_thresh)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--model_path",
    type=str,
    default="",
    help="Detector checkpoint path",
)
parser.add_argument("--config_path", type=str, default="", help="Detector config path")
parser.add_argument("--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'")
parser.add_argument("--max_track_distance", type=int, default="30", help="Maximum tracker point distance")
parser.add_argument("--track_points", type=str, default="bbox", help="Track points: 'centroid' or 'bbox'")
parser.add_argument("--img_scale", type=int, default="720", help="MMDetection inference size (pixels)")
parser.add_argument("--conf_thresh", type=float, default="0.35", help="MMDetection inference confidence threshold")
args = parser.parse_args()

detector = Detector(
    model_path=args.model_path, config_path=args.config_path, device=args.device, track_points=args.track_points
)

for input_path in args.files:
    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=args.max_track_distance,
    )

    for frame in video:
        detections = detector(frame, img_scale=args.img_scale, conf_thresh=args.conf_thresh)
        tracked_objects = tracker.update(detections=detections)
        if args.track_points == "centroid":
            norfair.draw_points(frame, detections)
        elif args.track_points == "bbox":
            norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
