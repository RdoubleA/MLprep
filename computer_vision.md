# Computer vision

- [Edge detection](#edge-detection)
- [Landmark detection](#landmark-detection)
- [Object detection](#object-detection)
- [Object tracking](#object-tracking)
- [Segmentation](#segmentation)
- [3D pose estimation](#3d-pose-estimation)
- [Face detection](#face-detection)
- [Style transfer](#style-transfer)


## Edge detection
A convolutional filter that can detect a vertical edge will have a column of 1s followed by a column of 0x and a column of -1s. This detects the contrast between columns in pixel intensities in an image. For horizontal edges, you have a row of 1s, 0s, -1s

## Landmark detection
Use a CNN to output the probability of an object you are trying to find plus the point locations of all the landmarks. For example, if you are detecting facial features to design a SnapChat filter, you will train a network to first identify whether an image has a face or not and also output values for landmark points around the eyes, the nose, the chin, ears, etc. You can also use this for pose estimation.

## Object detection

#### Classification with localization
This is an extension of predicting the object from an image. Instead, we add a few more outputs to the target label. We have one element denoting the probability of if an object is in the image at all, four elements defining the center point, width, and height of a bounding box around the object, and the softmax probabilities of what type of object that is. Then use MSE to train the model.

#### YOLO algorithm
Divide image into a grid (3x3 for example). Each square in the grid will have a vector containing the probability that there's an object in the anchor box in the square, its four bouding box parameters, and the class probability of the object type. Each square will contain this vector for each anchor box you decide to use to detect overlapping images. So, if you split your image to a 3x3 grid, and you're using a horizontal rectangle anchor box and a vertical rectangle bounding box, then your target label will be a 3 x 3 x 2 x 8 vector. Once the network has run through the image, it will output a bounding box for each anchor box for each grid cell. Then, non-max suppression is run for each class, where low probability bounding boxes are removed, and the highest ones are kept and any boxes with large overlap with the highest ones are suppressed.

#### R-CNN
Proposes regions using segmentation / selective search. Then, runs AlexNet on each of the region proposals resized to a fixed size. SVMs are used after the FC layers to classify the object and regress the bounding box. Issues with this method are that it takes a long time to train (~2000 region proposals per image), cannot be implemented in real time, and selective search is not a learning algorithm, so we can have bad region proposals.

#### Fast R-CNN
Fast R-CNN still uses selective search for region proposals. However, these are used after the CNN has run the image through instead of before, so we don't have to run it 2000 times per image. Now, the region proposals are overlaid on the feature map from the CNN, and ROI pooling is applied, whiich applies max pooling to resize the dimensions of the region to a fixed size. So, the region is partitioned to match the predefined fixed size you want, and each cell is maxpooled. Then, after ROI pooling, we use fully connected layers to classify objects and determine bounding boxes.

#### Faster R-CNN
Instead of using a region proposal method, we can learn that ourselves with another CNN. Faster R-CNN implements a regional proposal network that trains on the feature map of the feature extractor CNN to make region proposals.

Data augmentation - the bounding boxes, region proposals must also be adjusted

#### Mask R-CNN
This is the state-of-the-art model for instance segmentation, when objects are segmented and classified individually and similar objects are not lumped together. This model adds additional convolutional layers to Faster R-CNN to output a mask of probabilities for the segmentation of multiple objects.

It also adds ROI align, which is a variation of ROI pooling except it uses interpolation to help resize proposals from the region proposal network, leading to higher accuracy of segmented maps

#### ROI pooling vs Spatial Pyramid Pooling vs ROI align
Spatial Pyramid Pooling is a unique pooling method that warps the different sizes of the region proposals into a fixed size. It is a primary contribution of the SPPNet. It solves the problem of inputting region proposals of different sizes into a fully connected layer for classification in object detection tasks. It partitions the input activation into a grid based on the desired output dimension, then performs max pooling to downsample all the pixels in each grid element. SPP does this at multiple scales to get a fixed length vector for each region proposal.

ROI pooling is the same as SPP, except only one scale / level is used.

ROI align solves the problem of partitioning the input activation into a fixed grid. Each grid element will have different numbers of pixels because the dimensions will not always divide evenly. ROI align adds bilinear interpolation to solve this.

Read more at: https://jhui.github.io/2017/03/15/Fast-R-CNN-and-Faster-R-CNN/

https://d2l.ai/chapter_computer-vision/rcnn.html

https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd

https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272

## Object tracking

Object tracking is an extension of object detection for videos instead of images. It first requires an underlying object detection algorithm / network. Then, the next challenge is how to figure out which bounding boxes in the next frame correspond to which boxes in the previous frame. You can achieve this many ways:
- Compute the centroids of every box in every frame, then compute distances with every pair of frames. The box from the previous frame with the smallest distance is likely the same object.
- Compute the IoU between boxes in next frame with previous frame. The box in previous frame with highest IoU is the same object
- Use a deep learning model to extract features from every box. The box in the previous frame with the most similar extracted features is the same object. This is the deep sort algorithm.

## Segmentation

The output of the model is a probability map where every pixel is labeled with a probability of belonging to a certain type of class. Types of models that can achieve this are encoder - decoder networks like the UNet, or fully convolutional networks, FCN, that go directly from images to segmentation map.

Semantic segmentation labels all instances of the same object / category as the same class. For example, labeling all pixels in a medical image that are cancerous tissue. U-Net and other FCNs are commonly used for this task.

Instance segmentation labels instances of the same object / category as different items of the same class. This usually involves an additional task of object detection. Mask R-CNN is the most established model for this task.

## 3D pose estimation

OpenPose - Use two branches of a CNN to detect confidence maps for join locations and affinity maps for links between joints.

## Face detection

#### Face verification
One-shot learning is training a network to recognize one example. For instance, Face-ID has to verify your face, but how do you train that without 1 million images of your face?? This is very difficult for deep learning models, so we need to frame the problem differently. 

In the case of facial verification, we want the network to learn an embedding such that the distance from the embeddings of two images of the same person is small and it is large for embedding of two images of different people. Thus, we use the triplet loss to train a network to learn this embedding, where we have a dataset of multiple images of multiple people and we show the network an anchor image of the person we want to identify, a positive image containing the same person but in a different image, and a negative image containiing a different person. So the dataset must have multiple images of the same people. However, if trained properly, you can apply this network to the one shot learning problem and apply it to a totally new person with just one image since the embedding it has learned should be robust.

## Style transfer

Content loss - difference between features in intermediate layers of a CNN such as VGG

Style loss - difference between Gram matrices in intermediate layers of a CNN such as VGG. The gram matrix is the dot product of channel vectors and captures correlations between channels to capture the "style" of the image.

You can use GANs to achieve this, or you can compute the gradients with respect to the pixels themselves and update them
