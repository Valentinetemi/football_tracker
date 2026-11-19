# Third-party credits

This project uses the following open-source projects and pretrained-model tooling:

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO model loading and
  inference. Ultralytics is offered under AGPL-3.0 and an enterprise license; check the
  terms for your intended use.
- [Supervision](https://github.com/roboflow/supervision) by Roboflow for its ByteTrack
  integration and detection data structures (MIT).
- [ByteTrack](https://github.com/FoundationVision/ByteTrack), the multi-object tracking
  method by Zhang et al. (MIT).
- [OpenCV](https://opencv.org/) for video decoding, encoding, scene-change measurements,
  and annotations (Apache-2.0).
- [NumPy](https://numpy.org/) for numeric arrays (BSD-3-Clause).

The pipeline structure was informed by Roboflow's open-source
[Sports soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer),
which combines Ultralytics detections with Supervision ByteTrack. This repository's
implementation is intentionally smaller and written for this project; no model weights or
sample videos are redistributed here.

The optional `football-pitch-detection.pt` landmark checkpoint and its 32-point, 120×70 m
pitch configuration are published by that same Roboflow Sports example. This project reuses
the indexed geometry needed to interpret the checkpoint, then adds its own RANSAC quality
gates, cut invalidation, tactical renderer, and fail-closed behaviour. The checkpoint remains
Git-ignored and is not redistributed by this repository.
