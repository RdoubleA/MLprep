# Natural Language Processing

- [Word embeddings](#word-embeddings)
- [Topic modeling](#topic-modeling)
- [Sequence models](#sequence-models)
- [Large language models](#large-language-models)
    * [Transformers](#transformers)
    * [Attention](#attention)
    * [BERT](#bert)
    * [GPT3](#gpt-3)
    * [T5](#t5)
    * [LLaMA](#llama)
    * [Alpaca](#alpaca)
    * [Dolly 2.0](#dolly-20)



## Word embeddings

### TF-IDF
Instead of using word count, we want to incorporate the fact that some words are present in most documents but don't contain any information content (such as "the", "and"). The term frequency - inverse document frequency is a score that multiplies how often a word occurs in a document (tf) with the log of the number of total documents divided by the number of documents with the word (idf). IDF is close to 0 if the word is very common, and a very high number if the word is only present in the current document.

### Bag-of-words
Create a dictionary of all the words you'll see, and for each text count the number of each word. Those are your vectors. Very simple, inexpensive model that can be used for text classification, but doesn't preserve any information on context.

### Continuous bag-of-words
Define a window size for the words to be considered your context words for the target. For every target word, use one-hot encoded vectors for all your context words and feed into softmax regression model (single layer network with no hidden activation) to predict the probability of the target word. Trained weights are your embedding, and contain some context information.

### Skipgram
The inverse of CBOW. Instead of multiple word inputs and one word output you have the singular target word input and multiple context word outputs. You predict the context words given the target word. Weights are your embedding.

### Word2vec
Actually not a singular model, but a family of models that try to create embeddings using local context. The two architectures for doing so are Skipgram and CBOW. The second paper on word2vec proposes some training modifications for improved performance, such as subsampling (using less samples) high frequency words such as "the", and negative sampling (for all words in the output that should output 0 probability, only select some of them for weight updates instead of all of them to reduce run time. Positive labels are still updated normally).

### Byte-pair encoding
Not necessarily a word embedding, but rather a post-processing step after tokenization. Now you can iterate through the whole corpus and find the most common character pair. "er" for example. This becomes a new token that you add to the character corpus. Repeat again and continue to merge the most common character pair, until you get to whole words. The final corpus is now your tokenization dictionary, and each token is assigned a token ID based on how frequently it appears. This is then embedded using a simple linear layer that is learned.

BPE has some advantages, including being able to handle out-of-vocabulary problems by breaking down new words into familiar tokens, as well as being able to capture some sematic meaning. For example, it is likely that "-ing", "-ed", and "un-" could be their own tokens.

## Topic Modeling

### Latent semantic analysis
Create a document-term matrix, where rows are the documents and columns are the terms. The elements in the matrix are the tf-idf scores. Perform SVD on this matrix. This will yield a document-topic matrix, the singular values for each topic, and a topic-term matrix. You can use these to determine what topics are in which documents and which words were used for each topic.

### Latent dirichlet allocation
Similar to LSA, SVD, and matrix factorization in that it calculates a document-topic matrix and a topic-term matrix, except it represents this as a probability distribution instead of real values. LDA uses the dirichlet distribution, which returns vectors of probabilities that sum to 1. It is different from a multivariate Gaussian in that the values are probabilities instead of real values. It can be interpreted as the probability that a term belongs to a topic out of all the topics. 

One of the main advantages of LDA is its ability to generate human interpretable topics and to classify new documents based on the learned topics. One of its main disadvantages is that the number of topics K has to be set in advance and it assumes that the order of the words in the document does not matter

## Sequence models
A full discussion of these models can be found in the [deep learning guide](https://github.com/RdoubleA/MLprep/blob/master/deep_learning.md#sequence-models).

In brief:
- RNNs keep track of a hidden state that is propagated through timesteps, and modified by input and previous hidden state. Unfortunately, it is Markovian in the sense that current state only depends on prior state, so long-term dependencies are very difficult to learn. Also, vanishing and exploding gradients due to BPTT, and lack of parallelization make this an inefficient model.
- GRUs strive to improve on this by selectively updating the hidden state via two more gates, reset and update. Reset uses previous hidden state and input to determine how much of new hidden state will be changed by past hidden state. Update lets you weigh old or new hidden state more in the final new hidden state. It improved on issues with RNNs but still not enough
- LSTM explicitly modeling long-term dependencies by adding a memory state in addition to hidden, which is controlled by three gates: forget, input, output. _Forget gate_ determines how much of past memory to retain. _Input gate_ determines how much of input and past hidden state is added to memory. _Output gate_ determines the next hidden state by applying a filter on the updated memory. 

## Large language models

### Transformers
The key contribution of transformers is self-attention, which allows it to take in the entire sequence at once. This enables parallelization of computation, which makes training transformers significantly faster than RNN/LSTM/GRU. Additionally, it address our vanishing/exploding gradients problem as it no longer has to BPTT. Transformers have become very effective at classifying, generating, and translating text beyond its RNN counterparts and its become the current go-to model for text. It is also beginning to be applied to images as well. Transformers are a dense topic, so I refer you to the d2l page on [transformers](https://d2l.ai/chapter_attention-mechanisms/transformer.html) for details, images.

- First, the words are embedded via a word embedding (word2vec for example, or some other model). Typically, a combination of byte-pair encoding and a single layer neural network is used.
- Then, a positional encoding is added via a function. By processing sequentially, RNNs were able to maintain the positions of the words. Transformers achieve this with the positional encoding.
- Self-attention uses query, key, and value matrices to represent words. When you multiply the matrices, it represents the interactions between different words at different locations in the sequence and their strength/relevance. This enables the transformer to learn different grammatical relationships and context information when you stack many of these attention "filters".
- Residual connections and layer normalizations are added to promote gradient flow (to further prevent vanishing gradients). This is inspired from ResNet.
- Decoder uses a mask on future outputs so model doesn’t cheat when predicting next words.

### Attention
At a high level attention allows NLP models to place more weight on parts of the input text that are more relevant to the task at hand. For example, for sentiment classication maybe more weight will be needed on emotional valence words like "good" or "bad". Quite simply, they are weights on the input, but the way they are calculated is a little more involved. For a detailed discussion, section II of [this paper](https://arxiv.org/ftp/arxiv/papers/1902/1902.02181.pdf) is quite excellent.

Attention weights can be calculated many different ways, but most boil down to the query-key-value paradigm (in transformers as well).
- Query: the input elements that the attention mechanism gives emphasis to, this is the "task" that attention tries to focus on
- Key: the embedding vectors that you will use on which attention weights are computed (word2vec embeddings of each word in the input)
- Value: the embedding vectors you will apply the attention weights to, most often these correspond 1-to-1 with the keys and might even be the same embeddings, but they could also be different representations of the same word

The query and key are combined in an energy function f(q, k) to compute energy scores. Then they are transformed to a distribution via a distribution function (usually softmax) to compute the final weights. These are the attention weights that are applied to the values.

In transformers, self-attention uses the same word embeddings for query, key, and value matrices. This enables the self-attention head to learn complex relationships between words. These matrices are split among multiple "heads" to learn different relationships. The energy function is simply dividing by square root of query size and softmax is used to create a distribution. Then the resulting attention weights are applied to the value matrix. This is apply at multiple layers, so attention is also computed on subsequent transformations of the original input words. This [article](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853) goes into more detail.

### BERT
Bidirectional Encoded Representations from Transformers is a tranformer-based model from Google that is essentially a deep stack of transformer encoders. There are no decoders used in BERT. This model is focused on learning dense representations of words by using global context. It achieves this via "bidirectional" language tasks, or training tasks that involve the entire input sequence. BERT was trained using Masked Language Modeling (randomly mask some input words, place classification layer on top of encoded outputs, and predict masked words) and Next Sentence Prediction (take pairs of sentences and predict if the second is right after the first in the same document). Because of the unique way it is trained, BERT has the flexibility to be adapted for many NLP tasks. For example, you can use BERT for text classification by adding a simple two-layer neural network classifier on top of the encoded outputs of a pretrained BERT model and only train the NN weights. Actually in practice, the input texts are prepended with a CLS token, BERT output encodings for each token in the input, and only the output for the CLS token is fed into the NN classifier. If that's confusing, this Medium [article](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) goes into more detail.

### GPT-3
OpenAI's crowning achievement, the GPT model is a deep transformer that is trained on terabytes of internet text with roughly 175 billion parameters. The model was first pre-trained using an unsupervised learning task of next word prediction, then fine-tuned using a supervised learning task (predict a classification). Task conditioning is also implemented, meaning that you can feed the actual task to perform as natural language and GPT=3 learned to perform them. This enables its flexibility to be applied to any NLP task once you prompt it. This is the biggest advantage over BERT; GPT does not need fine-tuning for a specific task, it just needs a prompt and some examples (zero, one, or few-shot leearning) and it can deliver benchmark performance on that task, simply because the training data was so encompassing of all language. In fact it is so powerful that it is considered a big step towards artificial general intelligence.

The architecture of GPT-3 is a stack of transformer decoder layers, unlike BERT. Thus, it is primarily trained for text generation as opposed to text classification.

### T5
T5, short for "Text-to-Text Transfer Transformer", is a model architecture introduced by Google Research in 2019. Unlike previous Transformer-based models like BERT, T5 treats every NLP problem as a text-to-text problem, which allows it to train on a wide variety of tasks with a unified model architecture, training process, and hyperparameters. Other changes from BERT:
- The architecture is encoder-decoder. Thus, it is not trained with masked language modeling. It is trained similarly to denoising autoencoders, where random tokens are masked out and the decoder fills in these missing words, ex: "The cat [MASK] on the [MASK] [SENTINEL] The cat sat on the mat"
- It is fine-tuned on various tasks using task conditioning, wherre you prepend the task you want at the beginning

### LLaMA
LLaMA (Large Language Model Meta AI) is a family of transformer decoder only models released to the research community. The goal was to train models significantly smaller than GPT while still performant. It employs some improvements over other autoregressive LLMs:
- RMSNorm instead of LayerNorm, a simpler more efficient version. The paper contends that the re-centering invariance from LayerNorm can be dropped without hurting performance
- Rotary positional embeddings. See description in Dolly 2.0
- SwiGLU activation instead of ReLU, as in PaLM

### Alpaca

### Dolly 2.0
Dolly 2.0 is a 12B parameter model from the Pythia family of model. The [Pythia models](https://arxiv.org/abs/2304.01373) are a family of models spanning from small to large that, like GPT, are autoregressive transformer decoders. They are trained on a public dataset with next-word prediction and fine-tuned with high quality question answering datasets. There are a few key advancements that Dolly 2.0 implements that has shown to improve LLM training:
- _Flash attention_: employs low level optimizations, including fused CUDA attention kernels, and reduces the number of memory read/writes between GPU high-bandwidth memory and GPU on-chip SRAM to achieve 3x speed-up on GPT. It is now default in PyTorch. [Paper](https://arxiv.org/abs/2205.14135)
- _Rotary positional embeddings_: The original transformer uses an absolute positional encoding with the sinusoid, but some information is lost during the dot product attention, which only takes relative angle between two vectors into account. RoPE encodes absolute position by a rotation matrix, and modifies the word embedding vector by an angle multiple of its position. This ensures that dot product attention uses the absolute position information because the dot product takes angles into account. Thus you get both absolute and relative information. [Paper](https://arxiv.org/abs/2104.09864)
