# Computer vision

- [Edge detection](#edge-detection)
- [Landmark detection](#landmark-detection)
- [Object detection](#object-detection)
- [Object tracking](#object-tracking)
- [Segmentation](#segmentation)
- [3D pose estimation](#3d-pose-estimation)
- [Face detection](#face-detection)
- [Style transfer](#style-transfer)
- [Vision Transformer](#vision-transformer)
- [Video](#video)
    - [Multiscale Vision Transformer](#multiscale-vision-transformer)


## Edge detection
A convolutional filter that can detect a vertical edge will have a column of 1s followed by a column of 0s and a column of -1s. This detects the contrast between columns in pixel intensities in an image. For horizontal edges, you have a row of 1s, 0s, -1s

## Landmark detection
Use a CNN to output the probability of an object you are trying to find plus the point locations of all the landmarks. For example, if you are detecting facial features to design a SnapChat filter, you will train a network to first identify whether an image has a face or not and also output values for landmark points around the eyes, the nose, the chin, ears, etc. You can also use this for pose estimation.

## Object detection
This is an extension of predicting the object from an image. Instead, we add a few more outputs to the target label. We have one element denoting the probability of if an object is in the image at all, four elements defining the center point, width, and height of a bounding box around the object, and the softmax probabilities of what type of object that is. Then use MSE to train the model. Offline metrics that are used to evaluate models are intersection over union (IoU) of the bounding boxes or AUC of PR curve. How do you get precision and recall for bounding boxes? You can set IoU >= x as a true positive, IoU < x as a false positive, and IoU = 0 as a false negative (model completely missed a box). Then you can evaluate PR over various thresholds of x and calculate mean AUC for all object classes.

### YOLO algorithm
Divide image into a grid (3x3 for example). Each square in the grid will have a vector containing the probability that there's an object in the anchor box in the square, its bounding box parameters, and the class probability of the object type. Each square will contain this vector. Once the network has run through the image, it will output a bounding box for each detected object across the grid cells. Then, non-max suppression is run for each class, where low probability bounding boxes are removed, and the highest ones are kept and any boxes with large overlap with the highest ones are suppressed.

In YOLOv2, the model uses anchor boxes, which are predetermined boxes of different scales to handle irregular shapes, multiple objects, different objects of different sizes. Now each grid cell has outputs for each anchor box you decide to use. So, if you split your image to a 3x3 grid, and you're using a horizontal rectangle anchor box and a vertical rectangle anchor box, then your target label will be a 3 x 3 x 2 x 8 vector for a single class.

### R-CNN
Proposes regions using segmentation / selective search. Then, runs AlexNet on each of the region proposals resized to a fixed size. SVMs are used after the FC layers to classify the object and regress the bounding box. Issues with this method are that it takes a long time to train (about 2000 region proposals per image), cannot be implemented in real time, and selective search is not a learning algorithm, so we can have bad region proposals.

### Fast R-CNN
Fast R-CNN still uses selective search for region proposals. However, these are used after the CNN has run the image through instead of before, so we don't have to run it 2000 times per image. Now, the region proposals are overlaid on the feature map from the CNN, and ROI pooling is applied, whiich applies max pooling to resize the dimensions of the region to a fixed size. So, the region is partitioned to a grid to match the predefined fixed size you want, and each cell is maxpooled. Then, after ROI pooling, we use fully connected layers to classify objects and determine bounding boxes.

### Faster R-CNN
Instead of using a region proposal method, we can learn that ourselves with another CNN. Faster R-CNN implements a regional proposal network that trains on the feature map of the feature extractor CNN (usually ResNet) to make region proposals (object classification + bounding box). Then, ROI pooling is used on these proposed regions and fed into FC layers to refine the final object boxes.

Read more at: https://jhui.github.io/2017/03/15/Fast-R-CNN-and-Faster-R-CNN/

https://d2l.ai/chapter_computer-vision/rcnn.html

## Segmentation
The output of the model is a probability map where every pixel is labeled with a probability of belonging to a certain type of class. Types of models that can achieve this are encoder - decoder networks like the UNet, or fully convolutional networks, FCN, that go directly from images to segmentation map.

Semantic segmentation labels all instances of the same object / category as the same class. For example, labeling all pixels in a medical image that are cancerous tissue. U-Net and other FCNs are commonly used for this task.

Instance segmentation labels instances of the same object / category as different items of the same class. This usually involves an additional task of object detection. Mask R-CNN is the most established model for this task.

### Mask R-CNN
This is the state-of-the-art model for instance segmentation, when objects are segmented and classified individually and similar objects are not lumped together.

The model first uses a feature pyramid network, which uses ResNet and extracts feature maps at different scales. This is fed into the second stage, which is the region proposal network, which takes multiscale feature maps and proposes bounding boxes. Finally, there is ROI align, which is a variation of ROI pooling except it uses interpolation to help resize proposals from the region proposal network, leading to higher accuracy of segmented maps. The output of ROI align is fed into three parallel networks: classifier for object detection, bounding box regressor for localization, and a segmentation mask.

### ROI pooling vs Spatial Pyramid Pooling vs ROI align
Spatial Pyramid Pooling is a unique pooling method that warps the different sizes of the region proposals into a fixed size. It is a primary contribution of the SPPNet. It solves the problem of inputting region proposals of different sizes into a fully connected layer for classification in object detection tasks. Like ROI pooling, it partitions the input activation into a grid based on the desired output dimension, then performs max pooling to downsample all the pixels in each grid element. However, SPP does this at multiple scales to get a fixed length vector for each region proposal.

ROI pooling is the same as SPP, except only one scale / level is used.

ROI align solves the problem of partitioning the input activation into a fixed grid. Each grid element will have different numbers of pixels because the dimensions will not always divide evenly. ROI align adds bilinear interpolation to solve this.

https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd

https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272

## 3D pose estimation
OpenPose - Use two branches of a CNN to detect confidence maps for join locations and affinity maps for links between joints.

## Face detection/FaceNet
One-shot learning is training a network to recognize one example. For instance, Face-ID has to verify your face, but how do you train that without 1 million images of your face?? This is very difficult for deep learning models, so we need to frame the problem differently. 

In the case of facial verification, we want the network to learn an embedding such that the distance from the embeddings of two images of the same person is small and it is large for embedding of two images of different people. Thus, we use the triplet loss to train a network to learn this embedding, where we have a dataset of multiple images of multiple people and we show the network an anchor image of the person we want to identify, a positive image containing the same person but in a different image, and a negative image containiing a different person. So the dataset must have multiple images of the same people. However, if trained properly, you can apply this network to the one shot learning problem and apply it to a totally new person with just one image since the embedding it has learned should be robust.

## Style transfer
Content loss - difference between features in intermediate layers of a CNN such as VGG

Style loss - difference between Gram matrices in intermediate layers of a CNN such as VGG. The gram matrix is the dot product of channel vectors and captures correlations between channels to capture the "style" of the image.

You can use GANs to achieve this, or you can compute the gradients with respect to the pixels themselves and update them

## Vision transformer
ViTs are a natural extension of the transformer architecture to images. The primary advantage is that it is more scalable than CNNs. Similar to token/word embeddings in NLP, ViTs use patch embeddings to "tokenize" an image. An image is divided into a grid of patches and each patch is projected to an embedding vector (much like a 2D convolution). Then, the positional encoding is added. A cls token is appended for classification, as this token will attend to all patches in the image. The final embedding vectors are input into a multilayer transformer encoder that outputs a learned representation of each patch and the cls token. For classification, the cls vector is fed into a MLP to output a label. The patch representations can be used as features in some other downstream task. The transformer architecture itself is quite similar to standard language transformers, except a few extra dropouts and the use of GeLU instead of ReLU.

ViTs have outperformed ResNets by a significant margin and produce SOTA results, and as a result are often used as vision encoders to extract features from images.

Here is a succinct description of ViTs: https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html

## Video

### Object tracking
Object tracking is an extension of object detection for videos instead of images. It first requires an underlying object detection algorithm / network. Then, the next challenge is how to figure out which bounding boxes in the next frame correspond to which boxes in the previous frame. You can achieve this many ways:
- Compute the centroids of every box in every frame, then compute distances with every pair of frames. The box from the previous frame with the smallest distance is likely the same object.
- Compute the IoU between boxes in next frame with previous frame. The box in previous frame with highest IoU is the same object
- Use a deep learning model to extract features from every box. The box in the previous frame with the most similar extracted features is the same object. This is the deep sort algorithm.

### Multiscale vision transformer
ViTs work well on images but are computationally expensive on video, since the image resolution is maintained throughout the transformer layers. MViTs are a recent innovation that adds attention pooling in multihead attention to downscale the queries, keys, and values in each layer while simultaneously expanding the channel dimension. This results in a hierarchical learned representation of the images with less parameters. As a result, MViTs can be run more efficiently on video. It was also proven that MViTs can capture temporal information moreso than ViTs. The original [paper](https://arxiv.org/abs/2104.11227) goes into detail.
