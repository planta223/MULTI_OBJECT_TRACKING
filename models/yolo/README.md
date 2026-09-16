# YOLO segmentation weights

Place a custom Ultralytics instance-segmentation weight here, for example
`pipe_seg.pt`. Weight files are ignored by Git and must be copied separately
when moving the project to another PC.

A generic pretrained model does not know the project's custom Pipe class. Use
a segmentation model trained on the actual Pipe data; detection-only weights
that return bounding boxes without instance masks are rejected.
