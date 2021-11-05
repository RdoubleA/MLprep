# Natural Language Processing

- [Bag-of-words](#bag-of-words)
- [Skipgram](#skipgram)
- [Word2vec](#word2vec)
- [Transformers](#transformers)
- [Topic Modeling]


## TF-IDF
Instead of using word count, we want to incorporate the fact that some words are present in most documents but don't contain any information content (such as "the", "and"). The term frequency - inverse document frequency is a score that multiplies how often a word occurs in a document (tf) with the log of the number of total documents divided by the number of documents with the word (idf). IDF is close to 0 if the word is very common, and a very high number if the word is only present in the current document.

## Bag-of-words
Create a dictionary of all the words you'll see, and for each textt count the number of each word. Those are your vectors. Very simple, inexpensive model that can be used for text classification, but doesn't preserve any information on context

## Continuous bag-of-words
Define a window size for the words to be considered your context words for the target. For every target word, use one-hot encoded vectors for all your context words and feed into softmax regression model (single layer network with no hidden activation) to predict the probability of the target word. Trained weights are your embedding, and contain some context information

## Skipgram
The inverse of CBOW. Instead of multiple word inputs and one word output you have the singular target word input and multiple context word outputs. You predict the context words given the target word. Weights are your embedding.

## Word2vec
Actually not a singular model, but a family of models that try to create embeddings using local context. The two architectures for doing so are Skipgram and CBOW. The second paper on word2vec proposes some training modifications for improved performance, such as subsampling (using less samples) high frequency words such as "the", and negative sampling (for all words in the output that should output 0 probability, only select some of them for weight updates instead of all of them to reduce run time. Positive labels are still updated normally).

## Topic Modeling

### Latent semantic analysis
Create a document-term matrix, where rows are the documents and columns are the terms. The elements in the matrix are the tf-idf scores. Perform SVD on this matrix. This will yield a document-topic matrix, the singular values for each topic, and a topic-term matrix. You can use these to determine what topics are in which documents and which words were used for each topic.

### Latent dirichlet allocation