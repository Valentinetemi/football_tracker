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
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the optional general-purpose jersey
  text baseline and its published model weights (Apache-2.0). It is installed and loaded only
  for roster-enabled runs.

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

The jersey-number design also follows the task framing of the
[SoccerNet Jersey Number Recognition challenge](https://github.com/SoccerNet/sn-jersey):
number evidence belongs to a player tracklet, some tracks have no visible number, and distant
or blurred crops should remain unresolved. This project does not redistribute SoccerNet data
or models; the current swappable adapter uses EasyOCR and multi-frame voting as a modest
baseline.
