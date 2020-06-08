# Machine Learning Interview Prep

Explanations of major statistics and machine learning concepts that any aspiring data scientist / machine learning scientist must know for an interview. This is still under construction, but there's still a lot of good stuff here!

## Statistics

### Long-tailed Distribution

Long tailed distributions are probability distributions where a small number of values have very high frequency / probability and a large number of values have low frequency / probability, resulting in a long tail in the graph of the distribution that slowly tapers off.

Real world examples of this are social networks, where a majority of people have a relatively small number of a friends, and fewer and fewer people have a very large number of friends. Another examples is sales of different products, some products may be the most popular and drive a majority of the sales, whereas most of the products may have much fewer sales.

### Confidence Interval

Confidence intervals are a range of numbers that likely contains the population mean we are looking for. A 95% confidence interval means there is a 95% chance the interval contains the population mean (specifically, it means that the procedure used to create the interval, i.e., sampling and the math for making the interval, can create an interval with a 95% chance of containing the parameter)

Calculating the confidence interval involves the Central Limit Theorem. Because of CLT, our sample mean becomes a random variable that is normally distributed. Thus, we can calculate the range of values that will fall within 95% of the sampling distribution for our 95% confidence interval. This interval may contain the actual population mean with 95% confidence, since 95% of intervals we construct with this method will actually contain the population mean.

Z is the critical value of a standard normal distribution that lies at 95% of the area under the distribution, which is (-1.96, 1.96). If the population variance is known you can use that, where sigma is the STD of the population. If not, as in most cases, you will use the t-distribution and the sample standard error.

Some common mistakes:

95% confidence interval contains 95% of the population values – FALSE, 95% of the population values could be estimated from the sample. The 95% means that 95% of the intervals we make with this random sampling process will contain the population mean

95% confidence interval has a 95% chance of containing the population mean – kinda true, the fine print is that the procedure creates intervals that contain the population mean 95% of the time. An interval has the mean or it doesn’t

95% confidence interval of a larger sample size is more likely to contain the population mean than a smaller sample size – Nope, both intervals have an equivalent chance of containing the population mean since calculating confidence interval with standard error takes sample size into account. The main advantage of a larger sample size is narrowing that range.

This is a great resource for testing your understanding: http://www2.stat.duke.edu/~jerry/sta101/confidenceintervalsans.html

### Central Limit Theorem

Central limit theorem states that if you take an adequate number of samples from a population, regardless of the actual probability distribution of the populations, your sample means will approach a normal distribution, with a mean equivalent to the population mean and standard deviation according to standard deviation of population divided by square root of sample size. This means with repeated sampling from a population where the distribution is unknown, which is most real world distributions, we can calculate confidence intervals, conduct hypothesis testing, on the sample mean since it is normally distributed and make inferences about the population. This allows us to conduct t-tests with sample means to see if the population mean is nonzero, for example. At least 30 samples is considered an adequate number of samples for the means to approach normal distribution.

### Measures of Central Tendency

Mean – average of all examples
Advantages: all data points contribute to calculation, thus it is more sensitive when data points change or new points are added. Best used when distribution is data is near symmetric
Disadvantages: very sensitive to outliers, not ideal for long-tailed distributions for described the typical/middle value since it will be skewed towards outliers

Median – middle example when samples are arranged in order
Advantages: less sensitive to outliers, good for non-symmetric distributions (ex: income, house prices, where disparity is high)
Disadvantages: agnostic of distribution of data

Mode – most common element
Advantages: best used for categorical data where mean and median cannot be calculated
Disadvantages: not informative for continuous data


### Standard deviation vs standard error

Standard deviation describes the dispersion of the samples, and is agnostic of the distribution of samples or the population mean. Standard error measures how far the sample mean is from estimating the population mean. It is smaller than standard deviation, since it decreases as we increase number of samples (s / sqrt(n)). Standard error is used to construct confidence intervals, standard deviation is part of the calculation but it is not a metric itself in describing how accurate our estimates of the population mean are.

### Correlation vs Covariance

Both measure how a change in one variable affects the other, but correlation is normalized by the random variables standard deviation to range -1 to 1, while covariance is unbounded

### Type I vs Type II error

### Probability vs Likelihood

### Derive the MLE cost function for linear/logistic regression, when every observation has the same variance, and when every observation has a different variance

I point you to this comprehensive article on this topic: http://allenkunle.me/deriving-ml-cost-functions-part1

Know that minimizing the cost function for linear regression assumes the following:
1. Observations of X in the training set are independently, identically distributed
2. Residuals (errors from predicted to actual) are normally distributed with zero mean and the same variance for all examples


### Combinations and Permutations

Groups with the same elements but different orderings are **multiple different permutations** but are all the **same, one combination**

$$C=\frac{n!}{k!(n-k)!}$$

$$P=\frac{n!}{(n-k)!}$$

where $n$ is the number of objects you are choosing from, and you choose $k$ objects

Further reading: https://medium.com/i-math/combinations-permutations-fa7ac680f0ac

### Bayes Theorem

$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$

This is applied whenever we are given probabilities for certain conditions that are related. This is also the basis for the Naive Bayes classifier. This should be **memorized**

### Regression to the mean

Further observations following extreme values are closer to moderate values, i.e., if two parents are taller than average than their next child will likely be shorter than them, or closer than average. Of course, this assumes that the predominant factor is chance / luck.

### Probability Distributions

#### Normal/Gaussian

Also known as a bell curve, this is the most important distribution in statistics since many natural phenomena follow this curve. You should be able to call the percentage of values within each standard deviation from memory:

(put an image here)

Most important concept is that the conclusion of the central limit theorem is that sampling distributions approach a normal distribution. Many statistical tests such as t-tests assume a normal distribution of the samples, and machine learning algorithms also have these assumptions. For example, linear regression assumes the errors / residuals are normally distributed.

#### Poisson

This distribution is used for events in a time interval. The PDF looks like this:

$$P(X=k)=\frac{\lambda^ke^{-lambda}}{k!}$$

where $k$ is tthe number of events within an interval you are computing the probability of and $\lambda$ is the expected number of events to occur within the same interval. The mean and variance of this distribution is also $\lambda$. This video has great examples: https://www.youtube.com/watch?v=YWdOW29tdq8

#### Geometric

Use this distribution for problem where you want to find the chance of first success. The PDF looks like this:

$$P(X=x)=(1-p)^{x-1}p$$

Intuitively, this is simply calculating the probability of independent failure events up until the successful event given the p is the probability for any one success. The mean is $\frac{1}{p}$ and the variance is $\frac{1-p}{p^2}$ Here is a good video: https://www.youtube.com/watch?v=zq9Oz82iHf0&t=215s. Honestly this whole channel is great for understanding any distribution.

You should be able to solve problems such as this:

> You have a group of couples that decide to have children until they have their first girl, after which they stop having children. What is the expected gender ratio of the children that are born? What is the expected number of children each couple will have?

You can use the mean of the geometric distribution with $p=0.5$ to find the answer. You can use this mean to calculate the expected number of trials until a first success for any problem.

#### Binomial

Used when you want to find out the probability of a certain number of successes after a given number of events. Thus, we first must find the number of combinations with the desired number of successes and multiply by the probability of any one of those combination from happening. Thus:

$$P(X=x)={n \choose x}p^x(1-p)^{n-x}$$

where ${n \choose x}$ is the equation for combinations. The mean is $np$ and the variance is $np(1-p)$ An example:

> A balanced, six-sided die is rolled three times. What is the probability a 5 comes up exactly twice?

Here we let rolling a 5 be a "success" and anything else a "failure". Then, $n=3$ and $p=1/6$ and we should be able to calculate this probability.

#### Multinomial

This distribution is an extension of binomial to multiple classes instead of just the binary success or failure. The same concept of number of combinations times the probability of each of those outcomes applies:

$$P(X_1=x_1,...,X_k=x_k)=\frac{n!}{x_1!...x_k!}p_1^{x_1}...p_k^{x_k}$$

Each of $X_i$ is binomially distributed, so the mean and variance for each of $X_i$ is the same as the binomial distribution

> An urn contains 8 red balls, 3 yellow balls, and 9 white balls. 6 balls are randomly selected *with replacement.* What is the probability 2 are red, 1 is yellow, and 3 are white?

You should be able to do this using the multinomial distribution.

### Further reading

Probability cheatsheet: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiZuaWj1OvpAhVWs54KHQJkCwsQFjACegQIBBAB&url=https%3A%2F%2Fstatic1.squarespace.com%2Fstatic%2F54bf3241e4b0f0d81bf7ff36%2Ft%2F55e9494fe4b011aed10e48e5%2F1441352015658%2Fprobability_cheatsheet.pdf&usg=AOvVaw0ZN07X1cQMxVdnDVPPOzz-

Use this! https://github.com/JifuZhao/120-DS-Interview-Questions/blob/master/probability.md. Also this guide is really great in general.




## Machine Learning

### Decision Trees

Decision trees are a supervised learning algorithm that make multiple splits on a dataset to best separate all the samples into all the classification categories. It does so in a greedy manner at each decision to minimize heterogeneity. For classification, this can be information gain (a decrease in entropy) or minimizing Gini impurity, both of which are measures of heterogeneity. For regression, typically MSE from the mean of the split subset is minimized.

Effectively, decision trees is a nonlinear model by constructing multiple linear boundaries.

The advantages of decision trees is that they’re very easy to interpret, are good for feature selection by finding features that were most important for the splits, and require minimal preprocessing for the data. They can also handle categorical variables.

The disadvantages are that they overfit very easily. There are many methods to address this however. For example, pruning is used to remove leaf nodes that contribute little to information gain or classification accuracy. Random forests is an ensemble method that uses bagging with multiple decision trees to prevent overfitting. Actually, typically random forests are generally used over individual decision trees.

Gini impurity – measure of how likely it is that a sample will be classified incorrectly. Higher means more heterogenous with a max of 1.0, lower means more homogenous with a min of 0. It is calculated by:

So if we have categories that are evenly probable in a dataset, Gini impurity would be maximal. For binary classification, that would be 0.5 * 0.5 + 0.5 * 0.5 = 0.5. With more categories, this increases and approaches a value of 1.

Entropy – measure of uncertainty, or heterogeneity as used in decision trees. A high value indicates higher heterogeneity. Information gain is a decrease in entropy. Entropy involves log calculations, so gini impurity is typically used more.

### Random Forest

Random Forest is an ensemble of decision trees that can be used for classification or regression by aggregating the outputs of all the individual trees. There are two main features of Random Forests that make it more powerful than individual decision trees: 1) it uses bagging (bootstrapping aggregation), where the training dataset is resampled and each tree is trained on a different resample/bootstrap, 2) trees make decisions on a subset of features instead of all. Both of these features allow the individual trees to be more uncorrelated opinions, allowing them to cover the errors of the other and improving accuracy, reducing overfitting.

The most important hyperparameters for random forests are the number of trees and the number of features to use for each tree. Increasing the former usually improves performance upto a certain point, but increases computation time. It also increases model complexity and therefore variance. Increasing the latter increases variance and reduces bias, decreasing it will reduce variance and increase bias.

Increasing depth of each tree increases variance.

Increasing the minimum number of samples at each leaf node forces a regularization effect because it prevents the tree from becoming too complex. It increases bias.

https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d


### Accuracy Metrics

Accuracy is the total correct divided by all total classifications. Or, (TP + TN) / (TP + FP + FN + TN). Useful only when classes are balanced. If imbalanced, accuracy is not meaningful, i.e., detecting cancer that happens at a 1% rate in a population. You can just say no all the time and get a 99% accuracy.

Precision measures the proportion of predicted positives that were truly positive. That is, TP / (TP + FP). In the cancer example, we never predicted a true positive, thus precision is 0. Precision is useful when we want to be very sure of our prediction. For example, we don’t want to falsely diagnose someone with cancer, so we optimize for precision. However, since we’re taking extra care to be precise, we may let a lot of cases that actually have cancer slip by undetected, which could also be problematic. In other words, precision will maximize true positives, minimize false positives, but may not minimize false negatives. Depending on your problem, this may be ok.

Recall measure how many actual positives were correctly classified. It is useful when we want to make sure to capture all the positives. For example, if we are trying to predict terror threats, we want to capture all potential threats even if it may be a false alarm. However, if you only maximize recall, you can get a recall of 1 if you predict 1 for every example. Thus, recall will minimize false negatives and maximize true positives but may inflate false positives and decrease true negatives.

Precision and recall are two opposing forces. You want to be sure when you predict cases with cancer, but at the same time you want to be able to identify all cancer cases as much as possible, thus the tradeoff between precision and recall. F1 score manages this tradeoff, as it is a harmonic mean between the two. You can also use domain knowledge if you want to weigh finding all the positive cases more or being more confident in your positive classification more by the F1 beta score, which adds a factor to change the balance between the two.

https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226

Sensitivity is the true positive rate, or how many of the actual positives were correctly classified. It is equivalent to recall. Specificity is the true negative rate, or how many of the actual negatives were correctly classified.

Both of these come into play when using AUC ROC as a classification metric. When you vary the probability threshold used for classifying an example as positive, you change the model’s sensitivity and specificity. Increase the threshold, and you get fewer positives, which will decrease sensitivity and FPR and increase specificity. Decrease the threshold and you will get more positives, which will increase sensitivity and FPR and decrease specificity. This is plotted by an ROC curve, which plots sensitivity against FPR, or 1 – sensitivity, or the fraction of negatives that are incorrectly classified, over various thresholds. The area under this curve measures a model’s ability to separate classes. An AUC close to 1 indicates that the model is able to perfectly distinguish classes. An AUC close to 0.5 means the model cannot distinguish classes at all. An AUC of 0 means the model is completely reversing the classes. In multi-class problems, you compute ROC curves for one class vs all for each class. AUC is a great metric for general performance of a model irrespective of the threshold used. However, when the costs for false negatives or false positives are imbalanced, or the classes are heavily imbalanced, then AUC is not that informative. F1 score is more appropriate for imbalanced cases, or area under precision and recall curve.

https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

### Class Imbalance

When classes are heavily skewed (90% yes, 10% no), the model can have trouble learning the minority class and predicting future minority class examples. To address this, you can:
1)	Undersample the majority class and/or oversample the minority class (bootstrapping, repetition, etc)
2)	Weigh the minority class more in the cost function
3)	Use a more appropriate accuracy metric, such as AUC of precision-recall curve, F1 score

### Cross Validation

Cross validation is a validation method that partitions the dataset into folds, where all but one fold is used for training and the last fold is used for validation. This validation fold is rotated among all the folds and the model’s validation accuracy is averaged across all these experiments

### Feature Importance

In linear models, you can use the coefficients to compute feature importance. In tree-based models, more important features are closer to the root, so you can compute the average depth as an indicator of importance.

### MSE vs MAE

L1/MAE loss is not fully differentiable, so training may be less stable, but it is less sensitive to outliers since it gives equal weight to large error and small error. L2/MSE penalizes outliers more heavily, thus the model ends up becoming more sensitive to outliers. But it is fully differentiable and thus training is more stable. Both are generally pretty robust.

### Lasso vs Ridge regularization vs Elastic Net

Regularization shrinks the coefficients of a model and reduces overfitting. Lasso regularization/L1 enforces sparsity, as it is capable of driving weights to zero. Thus, it may even be used for feature selection. Good for when you know not all features will correlate with prediction. Ridge regularization/L2 ensures that no weights go to zero, but it is not robust to outliers unlike L1. It is good when all the features are important for the output. Elastic net uses both a linear and square term of the weights in the loss function and combines the advantages of both ridge and lasso.


### Bias vs Variance Tradeoff

Model error due to bias is a result of underfitting, or that the model is too simple and makes too many prior assumptions about the data to accurately fit it. Examples of high bias models are linear regression, since it assumes the data is linear, for example. If we increase model complexity we can find more complex patterns, but that leans towards overfitting the data and causes the model to generalize poorly. This is error due to variance, meaning the model is too complex and is fitting noise in the data instead of the general pattern. You can think of it as the model varies highly when changes datasets and generalizes poorly.

Thus, whether to increase model complexity or decrease model complexity is a tradeoff between bias and variance. High bias means better generalization but poorer fit, but high variance means excellent fit but poor generalization. The ideal model is somewhere in the middle.

You can determine whether a model has high bias or variance by examining the training and validation/test errors. A larger validation error than the training error indicates the model generalizes to unseen data poorly and is overfitting the training data, therefore it has high variance. If both training and validation error are high, then the model is underfitting the data and has high bias. There are many methods to address these issues:

Underfitting – expand the model capacity (add more trees, hidden layers, dimensionality, etc), add more input features, remove regularization

Overfitting – use regularization, decrease model capacity, use dropout (in neural networks), get more training data (helps model distinguish signal from noise and generalize to new data better), use ensemble methods, remove input features, early stopping

### Generative vs discriminative models

Discriminative models try to parametrize the posterior distribution p(y | x). That is, given a distribution of data, what is the most likely classification. It does not care about how the data was generated or what the distribution looks like. Alternatively, generative models attempt to parametrize p(x | y). That is, given samples of a data distribution, what is the underlying distribution that produces these. For example, a generative model shown images of cats would attempt to generate images that look similar to those cats, but discriminative models would try to classify all cat images as cats.

### Parametric models vs non-parametric models

Parametric models are defined by a constrained number of parameters that do not scale up with the size of the training set. On the other hand, nonparametric models increase number of parameters with more training samples. For example, linear regression, logistic regression, and SVMs are parametric models but decision trees, KNN are non-parametric models

### Linear Regression

Linear regression maps X to Y with a matrix of weights W and biases b. The weights are determined by minimizing the cost function of the predicted output vs the actual output. This is typically mean square error, but mean absolute error is also used sometimes. Gradient descent is used to minimize the cost function and find the optimal solution. To avoid overfitting, lasso or ridge or elastic net regularization is used, though ridge tends to outperform lasso. To add non-linearity, you can add polynomial and interaction terms.

### Logistic Regression

Take everything you know about linear regression, now add a sigmoid function at the end and use binary cross entropy for the cost function.

### Naive Bayes

Naive Bayes is a classifier that relies on Bayes rule to make predictions. Given a dataset, you can compute all the terms on the right hand side of the equation when a new sample is seen. It is a fast and easy to implement classifier that is used for sentiment analysis, recommendation systems, etc. However, it assumes that all features are independent of each other, which breaks down in most cases.

### K-nearest neighbors

KNN does not involve training a model. Instead, the entire dataset is kept and a new point is classified by calculating the Euclidean distance (or some other distance/similarity metric) to EVERY OTHER point and findg the K nearest neighbors, Then, the label is assigned based on the majority vote of those nearest neighbors. In regression, the mean of the neighbors labels are used. Since inference time requires the entire dataset and computing distances with the whole dataset, KNN is a computationally expensive model and is generally not used. 

When K is smaller, the model has high variance and tends to overfit. When K is larger, the model has high bias and tends to udnerfit.

### Support Vector Machines

Support vector machines are supervised learning models that attempt to find a hyperplane that separates two classes. In the vanilla SVM, it finds a linear hyperplane that maximizes the margins between the data points, so it doesn't just find any line that can separate the data points but one with the best distance between points that are close to the boundary and are difficult to classify. They are very similar to logistic regression models, the primary difference is the hinge loss instead of logistic loss and the kernel trick.

The kernel trick enables the flexibility of SVMs to handle classes that are not necessarily linearly separable. Perhaps the data is separable in a higher dimension, or some complex mapping from the current space to a higher-dimensional space. We could attempt to map the coordinates/features of each sample to a higher-dimensional space by, say, squaring some terms and adding interaction terms. But that would be very computationally expensive. Instead, kernel functions are functions that have already computed the dot product between two input vectors in high-dimensional space and have simplified it so that you don't have to do it yourself. Radial basis function, or RBF, is a popular kernel that is used in SVMs that essentially uses the distance from the origin or some point to the input vectors as the new embedding. In practice, the kernel function is applied to the entire dataset X and computes similarities between all the points in the higher-dimensional space. Why compute similarities between points? It helps capture the structure of the data by finding the relative distance between every point. But, the kernel trick allows you to leverage the use of the high-dimensional similarity wihtout having to do that mapping yourself. 

Hinge loss penalizes any sample the is either within the margin or outside the wrong margin linearly scaling with how far it is. It also requires that classes be 1 or -1 instead of the usual 0 and 1. But, once the class is outside the correct margin, there is no loss associated with it. This is unlike logistic loss, where there is a probabilistic interpretation of how likely a sample belongs to a class. So you can be above 0.5 probability for a class but still incur large errors if it's far from 1. This enables logistic loss the flexibility of having probabilistic interpretations, in cases where you want to know how likely a sample belongs in one class or multiple. Alternatively, SVM's hinge loss does not penalize samples once they are correctly classified. So, it does not have a probabilistic interpretation, but this tends to increase accuracy. 

Another difference between SVM and logistic regression is the ability of SVMs to handle non-linearly separable data. You could outfit a logistic regression model with square and interaction terms of the features, but computing all these yourself can get expensive. Since SVMs use the kernel trick, it is much better suited for these types of problems. However, it does not far well when there is overlap between classes and no clear margin can be found.

### Recommendation Systems

Recommendation systems are very important for many common services, such as suggesting movies you may like, or finding a product similar to the one you bought. It does this by gathering user-to-item relationship data and calculating similarities between users or between items to find the best new item for you.

User-to-item relationships can be represented in a matrix. The actual value could be explicit feedback, where the user physically rated an item, for example, or implicit feedback, which uses other metrics that imply the users satisfaction with an item, such as time spent watching a video. These can be represented as vectors for every given user, or for every given item. You can remove user biases by subtracting average user ratings from each rating.

Collaborative filtering uses the similarities between user vectors to predict a user's rating for a new item. Cosine similarity is typically used, then the top K users vectors are averaged, weighted by similarity, to compute a new vector for a new item for a user.

Content-based approaches typically require a lot of inherent information about the content itself independent of the user. The same similarity weighted sum can be used with these properties as the vectors for each item, and you sum the user's ratings for those items to predict how the user will like the new item.

These approaches sound straightforward, but the user-to-item matrix is typically very sparse, which cannot be handled well by these similarity metrics. It is also difficult to scale with more users and items since we have to compute similarities with each and every user/item. Instead, we can use something called matrix factorization.

The idea is to find a latent space that contains salient features for each user and each item that is hidden from our metrics. Once we have the latent vectors for both, we can compute the dot product of a user with an item to compute how likely the user will like that item. For example, one feature for a movie could be if it's a sci-fi movie, or for a user could be their preference for sci-fi movies, which we cannot measure directly but we can infer based on their ratings of movies. Typically this is done through singular value decomposition of the user-item matrix, which achieves exactly that, a latent embedding for both users and items ranked by importance. However, this only works for dense matrices. Since we have a very sparse matrix, we cannot do this analytically, we must approximate it with a cost function and use gradient descent. So we choose a certain number of features for user and item, then use mean square error to compute the difference between its dot product and the actual rating, then we using alternating least squares to optimize.

### The Curse of Dimensionality

When data dimensionality increases, data points become more equidistant from each other. This make its significantly more difficult to extract any meaningful structure from the data, especially for clustering methods. Additionally, searching the solution space with greater dimensions becomes computationally more complex and your model is prone to overfitting. More dimensinos means exponentially more space, meaning more space the model has to generalize to!

### Stochastic Gradient Descent vs Batch Gradient Descent

SGD uses one sample at a time to compute the gradient and make parameter updates.
- Slower since we cannot take speed benefits of vectorization of the whole dataset
- Uses much less RAM since we only need to store one sample at a time
- Noisier updates so it is less likely to get stuck at local minima
- Noisier updates so convergence may be slower

BGD uses the entire dataset to compute the gradient and make parameter updates.
- Can vectorize the gradient calculation of whole dataset, much faster to make updates
- Convergence is generally faster, especially good for convex functions
- Likely to get stuck as local minima
- Uses a significant amount of RAM to store the whole dataset

Minibatch GD is the compromise between the two that takes the speed bonus of vectorization and the low RAM costs of stochastic GD. Of course, batch size is still a hyperparameter that can be tuned.

## Deep Learning

### Activation Functions

Activation functions are necessary to introduce nonlinearities to neural networks to learn more complex functions and patterns. Otherwise, neural networks become a fancy linear regression.

Sigmoid – constrained output, fully differentiable, but tails can lead to vanishing gradients. Primarily used at final layer for binary classification

Tanh – constrained output, larger range for large gradients compared to sigmoid, but also has vanishing gradients

ReLU – computationally efficient, no vanishing gradient in linear range, may enforce a bit of sparsity. BUT, dying neurons when they enter negative regime. Most common activation function, this is a strong default option to use

Leaky ReLU – ReLU, except negative regime replaced by a slight slope, typically 0.01x. Fixes dying ReLU problem. Especially used in GANs where gradient flow is very important.

Softmax – sigmoid, generalized to multiple classes. Usually for numerical stability, we subtract the max of x from the exponent, which ensure you have large negative exponenets instead of large positive exponenets, which prevents overflow

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











