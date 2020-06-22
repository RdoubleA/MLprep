# Deep Learning

- [Activation Functions](#activation-functions)
- [What if all the weights are initialized with the same value?](#what-if-all-the-weights-are-initialized-with-the-same-value-)
- [What are the benefits of using a convolutional layer instead of a fully connected layer?](#what-are-the-benefits-of-using-a-convolutional-layer-instead-of-a-fully-connected-layer-)
- [How are RNNs trained?](#how-are-rnns-trained-)
- [Why are LSTMs preferred over vanilla RNNs?](#why-are-lstms-preferred-over-vanilla-rnns-)
- [Valid vs same padding](#valid-vs-same-padding)
- [Transfer learning](#transfer-learning)
- [What are exploding and vanishing gradients, and how do you deal with them?](#what-are-exploding-and-vanishing-gradients--and-how-do-you-deal-with-them-)
- [What are the advantages of batch normalization?](#what-are-the-advantages-of-batch-normalization-)
- [Optimizers and their benefits](#optimizers-and-their-benefits)
- [How do you deal with a small dataset?](#how-do-you-deal-with-a-small-dataset-)
- [What are GANs?](#what-are-gans-)
- [What is dropout?](#what-is-dropout-)
- [What is one-shot learning?](#what-is-one-shot-learning-)
- [Compute the receptive field of a node in a CNN](#compute-the-receptive-field-of-a-node-in-a-cnn)
- [Max vs average vs min pooling](#max-vs-average-vs-min-pooling)
- [Number of parameters in a convolutional layer](#number-of-parameters-in-a-convolutional-layer)
- [How do residual connections work?](#how-do-residual-connections-work-)
- [Why would you prefer many smaller filters over one large filter?](#why-would-you-prefer-many-smaller-filters-over-one-large-filter-)
- [Deep learning models](#deep-learning-models)
- [Model compression](#model-compression)

## Activation Functions

Activation functions are necessary to introduce nonlinearities to neural networks to learn more complex functions and patterns. Otherwise, neural networks become a fancy linear regression.

Sigmoid – constrained output, fully differentiable, but tails can lead to vanishing gradients. Primarily used at final layer for binary classification

Tanh – constrained output, larger range for large gradients compared to sigmoid, but also has vanishing gradients

ReLU – computationally efficient, no vanishing gradient in linear range, may enforce a bit of sparsity. BUT, dying neurons when they enter negative regime. Most common activation function, this is a strong default option to use

Leaky ReLU – ReLU, except negative regime replaced by a slight slope, typically 0.01x. Fixes dying ReLU problem. Especially used in GANs where gradient flow is very important.

Softmax – sigmoid, generalized to multiple classes. Usually for numerical stability, we subtract the max of x from the exponent, which ensure you have large negative exponents instead of large positive exponenets, which prevents overflow

## What if all the weights are initialized with the same value?

Then every neuron will output the same value, and the gradient updates to each neuron will be the exact same, meaning the network will not learn!

## What are the benefits of using a convolutional layer instead of a fully connected layer?

Captures spatial patterns. 

## How are RNNs trained?

## Why are LSTMs preferred over vanilla RNNs?

## Valid vs same padding
VALID padding means no padding, as it assumes the input volume dimensions are valid and no padding is needed.

SAME padding applies padding so that the filter size and stride can cover the whole image without skipping any data. It is called SAME because if you use a stride of 1, the output volume will have the same dimensions as the input regardless of filter size.

## Transfer learning

## What are exploding and vanishing gradients, and how do you deal with them?

## What are the advantages of batch normalization?

It directly addresses covariate shift, or when the distribution of activations changes from layer to layer.

Networks converge faster - by having layer activations with zero mean and unit variance, activations functions are used in their more useful regime. It helps promote gradient flow in this way.

The impact of weight initialization is less drastic since BN can refocus badly initialized layers.

It also adds a bit of noise between layers, acting as a form of regularization.

## Optimizers and their benefits

#### SGD
Vanilla stochastic gradient descent has the same learning rate for every parameter and does not adjust the learning rate. This can lead to slower convergence, since parameters that aren't as important are still moving the gradient equally to parameters that are more influential. In areas of the cost function where the hyperplane is steep, SGD does not go any faster then what the learning rate is fixed at. Thus, convergence is highly dependent on the learning rate

#### AdaGrad
Instead of using the same learning rate for every parameter, use different rates. Every step, the learning rate is divided by the size of all the past gradient updates. So if a parameter has been updated heavily, then the learning rate will slow down, and if it hasn't been updated much, learning rate is increased. However, this monotonically decreasing learning rate usually proves too aggressive and stops learning too early.

#### RMSprop
Adjusts AdaGrad by using a moving average update of the past few gradients for a less aggressive approach

#### Adam
Builds on RMSprop by computing moving average of past moving averages instead of past gradients themselves, which can be noisy. Also includes bias correction, which corrections for the first few time steps when there's no past gradient history to compute the moving average for, causing the steps to be biased at 0.

## How do you deal with a small dataset?

## What are GANs?

## What is dropout?

## What is one-shot learning?
See facial verification.

## Compute the receptive field of a node in a CNN

A neuron in a convolutional layer has a receptive field equivalent to the filter size and extended by the depth of the input volume. So if the input is an RGB image with 3 channels and the filter size is 5x5, each neuron has a receptive field of 5x5x3.

## Max vs average vs min pooling

All pooling downsamples activations without parameters. Average tends to smooth the image and remove high-frequency features. Max tends to select brighter parts of the image. Min tends to select darker parts of the image.

## Number of parameters in a convolutional layer

Number of parameters is the filter size * number of filters * input channel dimension + one bias per filter.

So 32 5x5 filters on an input volume with 16 channels is 5 * 5 * 16 * 32 + 32 parameters.

## How do residual connections work?

Residual connections are skip connections that bypass an entire layer and is added to the output of the same layer. This mean the layer is trrained to fit the residual of the mapping instead of the mapping directly. This has two major benefits:
1. Stacking more residual blocks and making the network deeper cannot degrade performance, as the layers can become identity blocks if needed. This allows deeper networks to be trained more effectively
2. Residual connections allow for gradient flow to bypass layers entirely, mitigating the vanishing gradients problem

## Why would you prefer many smaller filters over one large filter?

Stacking many smaller filters can create the same receptive field size as one larger filter, while simultaneously increasing nonlinearities and using less parameters.

## Deep learning models

#### LeNet
One of the first convolutional neural networks, it was used to classify digits on bank checks and spawned the famous MNIST dataset. It was revolutionary for it's novel use of convolutional layers to automatically learn important features in an image instead of using hand-egineered features as the rest of the field was doing. 

#### AlexNet
AlexNet won the ImageNet competition in 2012 by a large margin. It was the first to incorporate ReLU activation, dropout, data augmentation, and parallelized training on GPUs. It was deeper than LeNet and used more filters per layer.

#### GoogLeNet
It won the ImageNet challenge in 2014. Despite using 22 layers, much deeper than AlexNet, GoogLeNet uses significantly fewer parameters than AlexNet (60M to 4M). It achieves this with smaller filter sizes but more filters - which reduces parameters but increases non-linearities, 1x1 convolutions, and inception modules. Also uses batch normalization and RMSprop.

1x1 convolutions, dubbed "Network in a Network", and hence the "Inception" architecture, are essentially a multilayer perceptron of size equal to the number of filters applied to every element in the input volume. It is a clever way to stack additional nonlinearities to an input volume without changing the dimensions. It can also squash the filter dimension and reduce number computations in subsequent layers.

The inception module computes convolutions of multiple sizes and a pooling layer for the same input volume, then stacks each of those activations together. It simultaneously applies multiple convolutions to the same layer.

#### VGG
Runner-up to the ImageNet 2014 challenge, now de facto architecture to use to extract features. It's popular because of it's straightforward design. It also uses small receptive fields with many layers instead of fewer layers with large receptive fields. With 16 convolutional layers, it's actually much heavier than GoogLeNet and AlexNet, with 138M parameters.

#### ResNet
Uses skip connections to bypass layers so that you can created a deeper network without sacrificing gradient flow and risking vanishing gradients. Also gives the model the flexibility to use more complexity or not. This allowed the creators to use 152 layers but still have fewer parameters than VGG. It won first in ImageNet 2015 challenge.

## Model compression

Pruning - remove connections / weight with low absolute value or contribute little to objective function

Quantization - clustering / bundling weights that are close in value and representing them with one float value saves a lot of space

https://towardsdatascience.com/machine-learning-models-compression-and-quantization-simplified-a302ddf326f2

## Transformers

RNNs have been effectively at one-to-many, many-to-one, many-to-many sequence encodings. However, even with LSTMs and GRUs that add gating and memory, RNNs still have trouble retaining long sequence information and can only train on a sequence one item at a time, making them difficult to parallelize.

Transformers do not use RNNs to map a sequence to another. Instead, it uses multiple attention layers (multihead attention) to understand global context and encodes each words position to perform the mapping. This ends up being significantly faster and easier to train.


