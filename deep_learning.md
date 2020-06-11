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

### How are RNNs trained?

### Why are LSTMs preferred over vanilla RNNs?

### Valid vs same padding

### Transfer learning

### What are exploding and vanishing gradients, and how do you deal with them?

### What are the advantages of batch normalization?

### How do we perform searches of hyperparameters such as learning rate?

### Optimizers and their benefits

### How do you deal with a small dataset?

### What are GANs?

### What is dropout?

### What is one-shot learning?