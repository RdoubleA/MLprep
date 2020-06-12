## Computer vision

### Object detection

Two main stages - region proposal and classification of region. Region proposal outputs bouding boxes that contain the object in question, so the algorithm must output four continuous values, x and y of two corners of a rectangle (or corner and height and width), that contain an object. This can be done by selective search algorithm or with a CNN.

A CNN such as Fast R-CNN compresses an image and predict bounding boxes of multiple aspect ratios and sizes at different points/anchors in the image. If the box has over 0.7 overlap, or Intersection over Union with the ground truth bounding box then it is classified as correct.

Two losses are used: classification loss for if the box contains an object or not, and regression loss for the bounding box parameters, only if an object is in the box.

During test time, non maximum suppression is used to filter out the bounding box proposals. Proposals are ordered by confidence score. Each time a proposal is added to the final list, its IoU with every other proposal is calculated and any other proposal that has over 0.7 IoU is removed to avoid overlapping bounding boxes.

### Object tracking

Object tracking is an extension of object detection for videos instead of images. It first requires an underlying object detection algorithm / network. Then, the next challenge is how to figure out which bounding boxes in the next frame correspond to which boxes in the previous frame. You can achieve this many ways:
- Compute the centroids of every box in every frame, then compute distances with every pair of frames. The box from the previous frame with the smallest distance is likely the same object.
- Compute the IoU between boxes in next frame with previous frame. The box in previous frame with highest IoU is the same object
- Use a deep learning model to extract features from every box. The box in the previous frame with the most similar extracted features is the same object. This is the deep sort algorithm.

### Segmentation

The output of the model is a probability map where every pixel is labeled with a probability of belonging to a certain type of class. Types of models that can achieve this are encoder - decoder networks like the UNet, or fully convolutional networks, FCN, that go directly from images to segmentation map.

Semantic segmentation labels all instances of the same object / category as the same class. Instance segmentation labels instances of the same object / category as different items of the same class. This usually involves an additional task of object detection.

### 3D pose estimation

OpenPose - Use two branches of a CNN to detect confidence maps for join locations and affinity maps for links between joints.

### Face detection

Histogram of oriented gradients

### Style transfer

Content loss - difference between features in intermediate layers of a CNN such as VGG

Style loss - difference between Gram matrices in intermediate layers of a CNN such as VGG

You can use GANs to achieve this, or you can compute the gradients with respect to the pixels themselves and update them,
