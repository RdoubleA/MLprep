## Deep Learning

### Activation Functions

Activation functions are necessary to introduce nonlinearities to neural networks to learn more complex functions and patterns. Otherwise, neural networks become a fancy linear regression.

Sigmoid – constrained output, fully differentiable, but tails can lead to vanishing gradients. Primarily used at final layer for binary classification

Tanh – constrained output, larger range for large gradients compared to sigmoid, but also has vanishing gradients

ReLU – computationally efficient, no vanishing gradient in linear range, may enforce a bit of sparsity. BUT, dying neurons when they enter negative regime. Most common activation function, this is a strong default option to use

Leaky ReLU – ReLU, except negative regime replaced by a slight slope, typically 0.01x. Fixes dying ReLU problem. Especially used in GANs where gradient flow is very important.

Softmax – sigmoid, generalized to multiple classes. Usually for numerical stability, we subtract the max of x from the exponent, which ensure you have large negative exponents instead of large positive exponenets, which prevents overflow

### What if all the weights are initialized with the same value?

Then every neuron will output the same value, and the gradient updates to each neuron will be the exact same, meaning the network will not learn!

### What are the benefits of using a convolutional layer instead of a fully connected layer?

Captures spatial patterns. 

### How are RNNs trained?

### Why are LSTMs preferred over vanilla RNNs?

### Valid vs same padding

### Transfer learning

### What are exploding and vanishing gradients, and how do you deal with them?

### What are the advantages of batch normalization?

It directly addresses covariate shift, or when the distribution of activations changes from layer to layer.

Networks converge faster - by having layer activations with zero mean and unit variance, activations functions are used in their more useful regime. It helps promote gradient flow in this way.

The impact of weight initialization is less drastic since BN can refocus badly initialized layers.

It also adds a bit of noise between layers, acting as a form of regularization.

### Optimizers and their benefits

### How do you deal with a small dataset?

### What are GANs?

### What is dropout?

### What is one-shot learning?

### Compute the receptive field of a node in a CNN

A neuron in a convolutional layer has a receptive field equivalent to the filter size and extended by the depth of the input volume. So if the input is an RGB image with 3 channels and the filter size is 5x5, each neuron has a receptive field of 5x5x3.

### Max vs average vs min pooling

All pooling downsamples activations without parameters. Average tends to smooth the image and remove high-frequency features. Max tends to select brighter parts of the image. Min tends to select darker parts of the image.

### Number of parameters in a convolutional layer

### How do residual connections work?

### Why would you prefer many smaller filters over one large filter?

Stacking many smaller filters can create the same receptive field size as one larger filter, while simultaneously increasing nonlinearities and using less parameters.

### Deep learning models

#### LeNet

#### AlexNet

#### VGG

Uses small receptive fields with many layers instead of fewer layers with large receptive fields

#### GoogLeNet

Used the inception module

#### ResNet

Uses skip connections to bypass layers so that you can created a deeper network without sacrificing gradient flow and risking vanishing gradients. Also gives the model the flexibility to use more complexity or not.

### Model compression

Pruning - remove connections / weight with low absolute value or contribute little to objective function

Quantization - clustering / bundling weights that are close in value and representing them with one float value saves a lot of space

https://towardsdatascience.com/machine-learning-models-compression-and-quantization-simplified-a302ddf326f2


