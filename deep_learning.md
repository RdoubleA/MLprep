
# Deep Learning

This [resource](https://d2l.ai/chapter_preface/index.html) is like a free online textbook for deep learning and goes through all the concepts here in detail. I'll still briefly cover them here.

- [Neural networks](#neural-networks)
- [CNNs](#cnns)
- [Sequence models](#sequence-models)
- [Embeddings](#embeddings)
- [GANs and Adversarial ML](#gans-and-adversarial-ml)
- [Deep Learning Concepts](#deep-learning-concepts)


## Neural networks
A perceptron is a single neuron with threshold activation (outputs one if input is greater than 0, 0 otherwise). It essentially performs linear regression, where all the inputs are weighted and added with a bias. Unlike linear regression, a neuron applies a nonlinear activation function on this sum to determin the final output. There's a whole range of activation functions discussed below, but the most common used is ReLU due to its fast run time and simplicity in calculating gradients. The nonlinear activation function is the key feature of neural networks that distinguish it from other supervised learning models. Stack enough of these nonlinearities to make a neural network and provide enough data and NNs could theoretically approximate any function, or mapping from an input to an output. The model performance ceiling is very high, but run time is slower, a large amount of training data is required (1 - 100 million data points), training takes a long time, compared to similar models. If there was no activation function and only one neuron, it would just be linear regression. If the activation function was sigmoid with one neuron, it would be logistic regression. If there was no activation function and you used hinge loss with one neuron, it would be SVM.

Multilayer perceptrons are many layers of perceptrons stacked together, though it does not have to have threshold activation. It is more used as the general term for feedforward neural network. It learns by using gradient descent to minimize a loss function.

The course on CNNs at Stanford has great notes on [basics](https://cs231n.github.io/) of neural networks.

- [Activation functions](#activation-functions)
- [Backpropagation](#backpropagation)
- [Number of parameters](#number-of-parameters)
- [Weights initialization](#weights-initialization)
- [Batch normalization](#batch-normalization)
- [Dropout](#dropout)
- [Optimizers](#optimizers)

### Activation functions
The Stanford [notes](https://cs231n.github.io/neural-networks-1/#actfun) go into more detail.

**Sigmoid** - S shaped activation function that squashes output from 0 to 1. This is only used for the final output for binary classification. It is not used in hidden layers because of the vanishing gradients problem; towards the tail ends, gradients approach zero.

![](https://latex.codecogs.com/gif.latex?\sigma(x)&space;=&space;1&space;/&space;(1&space;&plus;&space;e^{-x}))

**Tanh** - S shaped activation function that squahes output from -1 to 1. This is generally used in the final output layers in GANs and autoencoders because of the output range. Tanh is zero-centered, unlike sigmoid, so activations that go into the next layer don't always start with a positive bias (which can hinder learning). However, you still have vanishing gradients, so this is generally not used.

![](https://latex.codecogs.com/gif.latex?\tanh(x)&space;=&space;2&space;\sigma(2x)&space;-1)

**ReLU** - essentially cuts off any values less than zero. This is the default activation function in hidden layers because of the speed of computation, and it greatly accelerates training, and no vanishing gradients in the positive region. However, the ReLU neurons can die in the negative region. This can usually be mitigated by setting the right learning rate though, so it's not too big an issue.

![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\max(0,&space;x))

**Leaky ReLU** - this addresses the dying ReLU problem by adding a small slope (typically 0.01) in the negative region. This is used in GANs where gradient flow is very crucial.

![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\mathbb{1}(x&space;<&space;0)&space;(\alpha&space;x)&space;&plus;&space;\mathbb{1}(x>=0)&space;(x))

**Softmax** – basically sigmoid, but generalized to multiple classes. Usually for numerical stability, we subtract the max of x from the exponent, which ensure you have large negative exponents instead of large positive exponenets, which prevents overflow. This is only used in final output layer for multiclass classification.

![](https://latex.codecogs.com/gif.latex?L_i&space;=&space;-\log\left(\frac{e^{f_{y_i}}}{&space;\sum_j&space;e^{f_j}&space;}\right))


### Backpropagation
You want to find the gradient of the loss function with respect to each weight value. However, the output has passed through several layers and several activation functions. Thus, you must use the chain rule to derive gradients for all the weights. This is why the activation function should be differentiable in order to compute gradients. Then the parameter updates flow "backwards" from the loss to all of the weights and biases to update them.

### Number of parameters
(size of hidden layer x size of inputs + 1 bias per neuron in hidden layer) * number of hidden layers

### Weights initialization
We must initialize our weights so that gradient flow is healthy in the beginning of training. Initializing with zeros does _not_ work because then all neurons will output the same value and receive the same gradient updates. Then, the network will learn nothing because all neurons are updating the same way. You can initialize with random values but this is not used in practice, because the output distribution of each neuron will be slightly different; the means could all be forced to be the same but the variance is actually a factor of how many inputs it receives. Thus, there are two common methods that are used in practice (more details in Stanford [notes](https://cs231n.github.io/neural-networks-2/#init)):

**He initialization** - samples from a normal distribution centered at 0 with variance sqrt(2 / n) where n is the number of input units. This is used by ReLU neurons (HeNormal in Keras).

**Xavier initialization** - a similar approach as He but reached conclusion that variance should be sqrt(2 / n_in + n_out). In practice, this is used for sigmoid layers (see this StackExchange [post](https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference) for comparisons). This is GlorotNormal in Keras.

### Batch normalization
Batch normalization is a differentiable layer in between the output of one hidden layer and the input to another that ensures activations have zero mean and unit variance. It directly addresses covariate shift, or when the distribution of activations changes from layer to layer. Networks converge faster - by having layer activations with zero mean and unit variance, activations functions are used in their more useful regime. It helps promote gradient flow in this way. The impact of weight initialization is less drastic since BN can refocus badly initialized layers. It also adds a bit of noise between layers, acting as a form of regularization. It is generally advised to include batch normalization between every layer.

### Dropout
Dropout is an additional regularization tool specific for neural network, on top of L2, L1 regularization which you can also use for NNs. At the output of every hidden layer, you can randomly zero some activations with a dropout probability p. This slows down training, but acts as a regularizer. During test time, we don't perform dropout but we have to multiply each neuron's output by the dropout probability to maintain the expected output value during training.

### Optimizers
Again, more details in the Stanford [notes](https://cs231n.github.io/neural-networks-3/#update). 

**Stochastic gradient descent** - simply modify each parameter by its gradient, factored by a global learning rate. Updates occur one sample at a time. One major drawback is that each parameter has the same learning rate, so tuning one learning rate for all weights can be ineffective.

```
# Vanilla update
x += - learning_rate * dx
```

**Batch gradient descent** - same as SGD, except updates occur with the entire dataset at a time.

**SGD vs BGD** - SGD uses one sample at a time to compute the gradient and make parameter updates.
- Slower since we cannot take speed benefits of vectorization of the whole dataset
- Uses much less RAM since we only need to store one sample at a time
- Noisier updates so it is less likely to get stuck at local minima
- Noisier updates so convergence may be slower

BGD uses the entire dataset to compute the gradient and make parameter updates.
- Can vectorize the gradient calculation of whole dataset, much faster to make updates
- Convergence is generally faster, especially good for convex functions
- Likely to get stuck as local minima
- Uses a significant amount of RAM to store the whole dataset

**Minibatch GD** is the compromise between the two that takes the speed bonus of vectorization and the low RAM costs of stochastic GD. Of course, batch size is still a hyperparameter that can be tuned. This method is still very sensitive to learning rate, and GD in general can get trapped in local minima.

**SGD with Nesterov Momentum** - Momentum is the idea that we can speed up gradient updates when we go in a direction consistently, such as downhill. Adding momentum should speed up optimization. Now, we have a "velocity" term which is the value that updates the parameters, and at each step the gradient can modify the velocity. The velocity is weighted by a momentum parameter, which is less than 1 to avoid parameter updates from accelerating indefinitely. Nesterov momentum adds the idea of evaluating the gradient at the future point instead of the current one. We know the current point will be updated by a velocity, so why not compute the gradient at that point to "look ahead" and speed up convergence?

```
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```


**AdaGrad** - Instead of using the same learning rate for every parameter, use different rates that are adaptively tuned as we learn so we can spend less time tuning it ourselves. Every step, the learning rate is divided by the sum of all the past gradient updates. So if a parameter has been updated heavily, then the learning rate will slow down, and if it hasn't been updated much, learning rate is increased. However, this monotonically decreasing learning rate usually proves too aggressive and stops learning too early.

```
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

**RMSprop** - Instead of normalizing by the sum of all past gradient updates which proves too aggressive, just use a moving average of past gradient updates. How much of the past updates to use is determined by decay_rate.

```
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

**Adam** - Builds on RMSprop by computing moving average of past moving averages instead of past gradients themselves, which can be noisy. Also includes bias correction, which corrects for the first few time steps when there's no past gradient history to compute the moving average for, causing the steps to be biased at 0. This is the default optimizer to use, although I've seen RMSprop or SGD w/ Nesterov be used.

```
# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x += - learning_rate * mt / (np.sqrt(vt) + eps)
```

## CNNs
Honestly, everything you need to know is in the Stanford [notes](https://cs231n.github.io/convolutional-networks/). I'll try to briefly summarize the key points here.

- [Convolutional layer advantages vs. fully connected layers](#convolutional-layer-advantages-vs-fully-connected-layers)
- [Number of parameters in a convolutional layer](#number-of-parameters-in-a-convolutional-layer)
- [Output dimensions of convolutional layer](#output-dimensions-of-convolutional-layer)
- [Receptive field](#receptive-field)
- [Dilated convolutions](#dilated-convolutions)
- [Pooling](#pooling)
- [Valid vs same padding](#valid-vs-same-padding)
- [Famous models](#famous-models)
  * [LeNet](#lenet)
  * [AlexNet](#alexnet)
  * [GoogLeNet](#googlenet)
  * [VGG](#vgg)
  * [ResNet](#resnet)

### Convolutional layer advantages vs. fully connected layers
CNNs introduce a key addition to a standard neural network: it uses convolutional layers instead of fully connected hidden layers to better handle images. The primary issue with using FC layers with images is that they don't scale well - you would need a massive number of parameters to apply weights and biases to every pixel in an image and have multiple layers of that. Convolutional layers contain a number of filters with a certain size that you scan or convolve across the width and height of the input volume, compute dot product with each step, and output that element in a new output volume with smaller dimensions (the Stanford notes has a great animation demonstrating this). This means you only need parameters for the filters, which is much smaller than the input volume. For this reason, convolutional layers are much lighter than their FC counterparts and scale much better. The scanning/convolution part also adds some translational invariance to the model, meaning objects can be in different positions in the image but the model can still learn to identify them.

### Number of parameters in a convolutional layer
F * F * D * K weights and K biases, where F is filter size, D is depth of input volume, K is number of filters. So only one bias per filter.

### Output dimensions of convolutional layer
The filter size, strides, and padding determine the dimensions of the output volume. The new width is W2 = (W1 - F + 2P)/S + 1, and the new height is calculated similarly. The depth is always the number of filters from the convolutional layer. F is filter size, W1 is original width before conv, P is padding amount, S is stride size.

### Receptive field
The area of an input volume that an output volume element corresponds to. This ends up just being the filter size.

### Dilated convolutions
Instead of applying a filter directly onto the image, you can skip a certain number of elements in the input between filter weights. This "dilates" the receptive field, enabling you to take advantage of a larger receptive field = more larger scale spatial information with the same computational costs and number of parameters. This [link](https://stackoverflow.com/questions/41178576/whats-the-use-of-dilated-convolutions) has a great explanation.

### Pooling
Pooling applies a convolution to the input but does not perform dot product. Instead, it will perform a simple operation such as MAX, MIN, AVG, or even L2 NORM without the use of any parameters. The purpose of the pooling layer is to progressively downsample the input and reduce its dimensions so that we can use less parameters in the future layers and reduce computation time. For this reason, it has a regularization effect. Recently, pooling has been falling out of favor and is being replaced by strided convolutions, which will convolve and do the downsampling together.

The default operation is MAX and tends to perform the best. Average tends to smooth the image and remove high-frequency features. Max tends to select brighter parts of the image. Min tends to select darker parts of the image. The output volume is determined the same way as in the convolutional layer, except there is no padding.

### Valid vs same padding
VALID - no padding, if filter goes beyond input then discard this operation. Some parts of image may be cut off if filter size doesn't divide evenly into image size. SAME - adds padding such that output volume is the same dimensions as the input. For odd amounts of padding, pad on right first. This [link](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t) explains more.

### Famous models
#### LeNet
One of the first convolutional neural networks, it was used to classify digits on bank checks and spawned the famous MNIST dataset. It was revolutionary for it's novel use of convolutional layers to automatically learn important features in an image instead of using hand-engineered features as the rest of the field was doing. 

#### AlexNet
AlexNet won the ImageNet competition in 2012 by a large margin. It was the first to incorporate ReLU activation, dropout, data augmentation, and parallelized training on GPUs. It was deeper than LeNet and used more filters per layer.

#### GoogLeNet
It won the ImageNet challenge in 2014. Despite using 22 layers, much deeper than AlexNet, GoogLeNet uses significantly fewer parameters than AlexNet (60M to 4M). It achieves this with smaller filter sizes but more filters - which reduces parameters but increases non-linearities - 1x1 convolutions, and inception modules. Also uses batch normalization and RMSprop.

**1x1 convolutions**, dubbed "Network in a Network", and hence the "Inception" architecture, are essentially a multilayer perceptron of size equal to the number of filters applied to every element in the input volume. It is a clever way to stack additional nonlinearities to an input volume without changing the dimensions. It can also squash the filter dimension and reduce number computations in subsequent layers.

The inception module computes convolutions of multiple sizes and a pooling layer for the same input volume, then stacks each of those activations together. It simultaneously applies multiple convolutions to the same layer.

#### VGG
Runner-up to the ImageNet 2014 challenge, now de facto architecture to use to extract features. It's popular because of it's straightforward design, and proves that going deeper is necessary for bettering performance. It also uses small filters with many of them instead of fewer filters with large receptive fields. With 16 convolutional layers, it's actually much heavier than GoogLeNet and AlexNet, with 138M parameters.

#### ResNet
Uses skip connections to bypass layers. This effectively lets the network decide to use a layer or not. Additionally, it promotes gradient flow because if gradients start vanishing they can backpropagate through the skip connections, so you can create a deeper network without worrying about gradients. This allowed the creators to use 152 layers but still have fewer parameters than VGG. It won first in ImageNet 2015 challenge.

## Sequence models
Neural networks with sequences model for the continuity or time steps in the data and the sequential relationships between input items, for example text data or waveforms. This sounds like you can use a CNN to slide across the input, but the drawback with CNNs is that it requires a fixed input size and fixed output size, but in sequences you might have variable input and output sizes. These models can perform one-to-one mapping (classifying single words), one-to-many (language generation, image captioning), many-to-one (text classification, action prediction from video frames), or many-to-many (translation, video captioning). You can [refer](http://cs231n.stanford.edu/slides/2021/lecture_10.pdf) to the slides from the Stanford course for more details about everything discussed here.

- [RNNs](#rnns)
  * [Backpropagation through time](#backpropagation-through-time)
- [GRUs](#grus)
- [LSTMs](#lstms)
- [Transformers](#transformers)

### RNNs
Recurrent neural networks are the simplest sequence model. The idea is that it maintains a hidden state, h, that is updated as the sequence progress, and the weights let you travel from one hidden state to another. 

![](https://latex.codecogs.com/gif.latex?h_t=tanh(W_{hh}h_{t-1}&plus;W_{xh}x_t)\newline&space;\indent&space;y_t=W_{hy}h_t)

In a way it's similar to a Markov model because the next state is dependent only on the previous. This is the primary drawback, as RNNs cannot learn large-scale relationships or grammatical context because a hidden state's information is barely propagated beyond the immediate next hidden state. Another drawback is vanishing and exploding gradients, which is a byproduct of BPTT. This makes it difficult to converge. A third drawback is that since RNNs process the input data sequentially, it cannot be parallelized. Thus, training is very slow and it limits how large you can make your model. 

You can stack recurrent layers and wire hidden states as inputs into the next layer to make _deep RNNs_. If you have two layers and you reverse the direction that the sequence is processed in one of the layers, then you have a _bidirectional RNN_.

#### Backpropagation through time
When you derive the gradient for backpropagation through time, the tricky term is the partial derivative of h_t with respect to W_h, since it depends on previous hidden states h_t-1, which also depends on W_h and more hidden states. Eventually you get a nasty product of gradients, which explains why they can vanish or explode. Typically, **gradients are clipped** to prevent exploding gradients, but they can still vanish. This [link](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html) shows the full derivation.

![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;h_t}{\partial&space;w_h}=\frac{\partial&space;f(x_{t},h_{t-1},w_h)}{\partial&space;w_h}&plus;\sum_{i=1}^{t-1}\left(\prod_{j=i&plus;1}^{t}&space;\frac{\partial&space;f(x_{j},h_{j-1},w_h)}{\partial&space;h_{j-1}}&space;\right)&space;\frac{\partial&space;f(x_{i},h_{i-1},w_h)}{\partial&space;w_h}.)

### GRUs
The main issue with RNNs is retaining long-term dependencies, but this isn't explicitly modeled. Gated Recurrent Units address this by engineering two gates that can control how much of the new hidden state is influenced by the old and by the input. More details in the [d2l site](https://d2l.ai/chapter_recurrent-modern/gru.html) which contains helpful visualizations.

The reset gate and update gate both control how much the old hidden state influences the new one, but through different ways. The reset gate is dependent on the input and past hidden state. It is dot producted with the hidden state when determining the new state to determine how much of the past hidden state we want to reset and how much we want it to influence the new hidden state. The update gate is also dependent on the input and past hidden state, but it is used to weight either the old hidden state more or the incoming new hidden state more.

![](https://latex.codecogs.com/gif.latex?\mathbf{R}_t&space;=&space;\sigma(\mathbf{X}_t&space;\mathbf{W}_{xr}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{hr}&space;&plus;&space;\mathbf{b}_r)\newline&space;\indent\mathbf{Z}_t&space;=&space;\sigma(\mathbf{X}_t&space;\mathbf{W}_{xz}&space;&plus;&space;\mathbf{H}_{t-1}&space;\mathbf{W}_{hz}&space;&plus;&space;\mathbf{b}_z)\newline&space;\indent\tilde{\mathbf{H}}_t&space;=&space;\tanh(\mathbf{X}_t&space;\mathbf{W}_{xh}&space;&plus;&space;\left(\mathbf{R}_t&space;\odot&space;\mathbf{H}_{t-1}\right)&space;\mathbf{W}_{hh}&space;&plus;&space;\mathbf{b}_h)\newline&space;\indent\mathbf{H}_t&space;=&space;\mathbf{Z}_t&space;\odot&space;\mathbf{H}_{t-1}&space;&plus;&space;(1&space;-&space;\mathbf{Z}_t)&space;\odot&space;\tilde{\mathbf{H}}_t)

If R is close to 1, the candidate hidden state is calculated just like in a vanilla RNN. If R is close to 0, it is calculated just like a NN. If Z (update gate) is close to 1, then we carry more of the past hidden state forward and don't update it. If Z is close to 0, then we favor the new update and forget past memory.

### LSTMs
Long Short Term Memory models try to address the same problem in RNNs that GRUs tried to address - how to retain memory from past time steps. While GRUs regulated the hidden state with the reset and update gates, LSTMs add another "thread" that's carried across time steps in addition to hidden state that's called memory. LSTMs regulate the memory via forget, input, and output gates. These are all dependent on the current input and past hidden state much like the reset and update gates in GRUs.
- _Forget gate_: determines how much of past memory to retain
- _Input gate_: determines how much of input and past hidden state is added to memory
- _Output gate_: determines the next hidden state by applying a filter on the updated memory

The [d2l page](https://d2l.ai/chapter_recurrent-modern/lstm.html) on LSTMs provides helpful visualizations.

### Transformers
The key contribution of transformers is self-attention, which allows it to take in the entire sequence at once. This enables parallelization of computation, which makes training transformers significantly faster than RNN/LSTM/GRU. Additionally, it address our vanishing/exploding gradients problem as it no longer has to BPTT. Transformers have become very effective at classifying, generating, and translating text beyond its RNN counterparts and its become the current go-to model for text. It is also beginning to be applied to images as well. Transformers are a dense topic, so I refer you to the d2l page on [transformers](https://d2l.ai/chapter_attention-mechanisms/transformer.html) for details, images.

- First, the words are embedded via a word embedding (word2vec for example, or some other model). 
- Then, a positional encoding is added via a function. By processing sequentially, RNNs were able to maintain the positions of the words. Transformers achieve this with the positional encoding.
- Self-attention uses query, key, and value matrices to represent words. When you multiply the matrices, it represents the interactions between different words at different locations in the sequence and their strength/relevance. This enables the transformer to learn different grammatical relationships and context information when you stack many of these attention "filters".
- Residual connections and layer normalizations are added to promote gradient flow (to further prevent vanishing gradients). This is inspired from ResNet.
- Decoder uses a mask on future outputs so model doesn’t cheat when predicting next words.

## Embeddings
### Two-tower network
This is used when you need to compute similarity between two embeddings, usually in ranking scenarios such as candidate generation for recommending movies to users, or posts to users. Essentially we have one encoder network (hidden layers get progressively smaller) for each embedding. The encoders output an embedding vector. The loss function to optimize is the different between the dot product similarity of these vectors and the actual feedback label which indicates the relevance between the two items (users and movies, etc). Once you've trained the network, you don't use this directly for predictions but you can generate the latent vectors for the two items. Then, you can take the K nearest neighbors in the embedding space to suggest items that are most likely to be relevant. 
### Triplet loss and one shot learning
The idea of triplet loss is to train an encoder network such that it places points that are of the same class closer together and of different classes further apart by at least a margin. You need three samples for each forward pass: one as the class of interest (anchor), one sample that's of the same class (positive), and one that's different (negative). Ideally, you mine triplets with semi-hard negatives, meaning the difference of the distance between anchor and positive and distance between anchor and negative is less than the margin. Andrew Ng's [lecture](https://youtu.be/d2XB5-tuCWU) on this topic is useful.

Triplet loss is used for **one-shot learning**, when you want to learn a new class with only one sample. Triplet loss learns a robust embedding such that when you learn a new class with one sample, any new samples of that class the distances to the original will just be computed to classify it. This is useful for facial verification, where you don't have the ability to train on thousands of images of a new person's face just to enable face ID.

#### Siamese network
Twin (or more) networks that are used to compute similarity between two images. They are twins because they use the same weights and they are updated with the same gradients. Actually, it's just one network in practice, but you forward pass two inputs before computing loss and backpropagating. You would use a similar setup for triplet loss and one shot learning.

### Autoencoders
Autoencoders consist of two networks: an encoder and decoder. An encoder progressively compresses the output volume and decoder networks progressively grow the output volume. It is self-supervised - meaning there is no output label. The output is actually the input - the encoder learns to compress the input to an embedding space such that the decoder can reconstruct the same input from that embedding. Thus, it's a way to learn a feature representation of an image, text, etc.

What's used in practice is the **variational autoencoder**. Instead of a fixed embedding vector, the encoder learns parameters of a probability distribution (typically Gaussian) and the decoder learns how to sample this correctly. This also let's you use the VAE as a generative model, as you can detach the encoder and sample from the latent space to generate samples.

## GANs
Generative Adversarial Networks are just that - a self-supervised generative model that learns through adversarial means. It is adversarial because two submodels are pitted against each other. The **generator** is the model we are interested in training and learns the distribution of the data by trial and error. It is similar to a decoder network from an autoencoder. The **discriminator** helps teach the generator by telling it if it's close to real data or not. The generator's learning task is to fool the discriminator by having it think the generated data was real. The discriminator's learning task is to successfully distinguish real and fake. The loss function is the minimax loss (described in detail [here](https://neptune.ai/blog/gan-loss-functions)) which is similar to binary cross-entropy of the real and fake distributions. In effect, the generator tries to maximize `log(D(G(z))` and the discriminator tries to maximize `log(1 - D(G(z)))`, which are competing objectives.

### Challenges with GANs
Gans are very difficult to train. The optimal steady state where both networks stabilize and perform the best they can do is difficult to achieve.
- Mode collapse: the generator "settles" for always producing one type of sample instead of a range of samples in the dataset because it seems to fool the generator sufficiently
- Vanishing gradients: if the discriminator gets too good, it will stop providing gradients to the generator
- Convergence: with minimax loss, converging during training is very sensitive to hyperparameters

Because of these challenges, there are many proposed alternatives to the vanilla GAN, such as conditional GAN, and Wasserstein GAN, which I won't go over here.

## Deep Learning Concepts

- [Transfer learning](#transfer-learning)
- [Exploding and vanishing gradients](#exploding-and-vanishing-gradients)
- [How do you train with a small dataset?](#how-do-you-train-with-a-small-dataset-)
- [Data augmentation](#data-augmentation)
- [Residual connections](#residual-connections)
- [Are many smaller filters better than one large filter?](#are-many-smaller-filters-better-than-one-large-filter-)
- [Model compression](#model-compression)

### Transfer learning
Transfer learning is when you take a model that was trained on a simlar task or on similar dataset and use it in your specific task while fine-tuning some parameters. This is the go to approach for most teams that have limited compute resources. For example, Google can train a transformer model from scratch with billions of samples in a reasonable amount of time to udnerstand language context and generate robust embeddings for any word. Instead of repeating that, we just use their pretrained model and fine-tune some of the layers for our specific task.

You have the option of freezing none to some to all layers based on how much training data you have. If you have limited data, then you might have to freeze most layers, and if you have plenty of data, you might only have to freeze a few. Or you can use the model's output as features for another task-specific model, which is the case for models such as word2vec that learn general word embeddings.

### Exploding and vanishing gradients
When gradients grow exponentially large during training or disappear. This can happen if the optimization algorithm requires a long product of gradients (such as in BPTT) or we enter a "dead" regime in the activation functions used (negative part of ReLU, flat parts of sigmoid and tanh). To address this, you can use a different activation function, or clip gradients, or use residual connections.

### How do you train with a small dataset?
Don't use deep learning, use a simpler model. If you must use NNs, then try data augmentation.

### Data augmentation
You can enhance your current dataset by applying some transformations to existing samples and expand the dataset. For example, adding white noise, translating or rotating images, adjusting brightness, etc. This can also help your model become invariant to these changes.

### Residual connections
Residual connections are skip connections that bypass an entire layer and is added to the output of the same layer. This mean the layer is trrained to fit the residual of the mapping instead of the mapping directly. This has two major benefits:
1. Stacking more residual blocks and making the network deeper cannot degrade performance, as the layers can become identity blocks if needed. This allows deeper networks to be trained more effectively
2. Residual connections allow for gradient flow to bypass layers entirely, mitigating the vanishing gradients problem

### Are many smaller filters better than one large filter?
Stacking many smaller filters can create the same receptive field size as one larger filter, while simultaneously increasing nonlinearities and using less parameters.

### Model compression
Pruning - remove connections / weight with low absolute value or contribute little to objective function

Quantization - clustering / bundling weights that are close in value and representing them with one float value saves a lot of space

https://towardsdatascience.com/machine-learning-models-compression-and-quantization-simplified-a302ddf326f2

### Can you input variable sized images into a CNN?
The fixed size input constraint on CNNs is solely due to any FC layers at the end, convolutional layers are agnostic of input size. So, you can make a fully convolutional network with no FC layers or use spatial pyramid pooling if you want to keep a FC layer. See more about SPP in the computer vision guide. Here is a [link](https://stats.stackexchange.com/questions/388859/is-it-possible-to-give-variable-sized-images-as-input-to-a-convolutional-neural) explaining more.







