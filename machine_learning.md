# Machine Learning

- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [ML Concepts](#ml-concepts)
- [Industry ML](#industry-ml)

## Supervised Learning
This category of ML models require labelled data where the outputs are known. The models then learn the mapping from input to output.

- [K-nearest neighbors](#k-nearest-neighbors)
- [Linear regression](#linear-regression)
- [Generative models](#generative-models)
  * [Gaussian/Linear discriminant analysis](#gaussian-linear-discriminant-analysis)
  * [Naive Bayes](#naive-bayes)
- [Discriminative models](#discriminative-models)
  * [Logistic regression](#logistic-regression)
  * [Support vector machines](#support-vector-machines)
  * [Perceptron/Linear classifier](#perceptron-linear-classifier)
- [Decision trees/CART](#decision-trees-cart)
- [Ensemble methods](#ensemble-methods)
  * [Random Forest](#random-forest)
  * [Boosting](#boosting)
- [Neural networks](#neural-networks)
- [Optimizers](#optimizers)
- [Model evaluation](#model-evaluation)
  * [Classification Metrics](#classification-metrics)
- [Loss functions](#loss-functions)
  * [MSE vs MAE](#mse-vs-mae)

### K-nearest neighbors
The intuition and assumption behind KNN is that data points that are clsoe together will likely have the same label / output. For a new data point, compute the distances to all other data points in the dataset. Take the K nearest points, and the majority label is the predicted label (average their values for regression). Because we have to keep the entire dataset, this model does not scale well (nonparametric) and is slow. It is not used in practice. In fact it's not really a model at all because it doesn't really learn anything, just computes distances.

When K is smaller, the model has high variance and tends to overfit. When K is larger, the model has high bias and tends to underfit.

### Linear regression
Let's derive the linear regression model, its loss function, and optimization from scratch. Linear regression maps X to Y with a matrix of weights theta and biases b, just like y = mx + b. Let theta be the weights we want our model to find, such that:

![](https://latex.codecogs.com/gif.latex?y&space;=&space;\theta^Tx&space;&plus;&space;\epsilon)

Here, we've added epsilon, which will captures unmodeled effects (biases) and random noise. Let's also assume this noise is normally distributed and is IID. This means the probability density function of epsilon is:

![](https://latex.codecogs.com/gif.latex?p(\epsilon)&space;=&space;\frac{1}{\sqrt{2\pi}\sigma}\exp{(-\frac{\epsilon^2}{2\sigma^2})})

Remember that this is the Gaussian distribution, and we're assuming all samples have the same known variance. This may not be true in all cases, especially in research experiments where each data point may come from a different source with a different unknown variance (such as in multiple regression, where there's multiple input variables). If the variances are unknown and different, then the distribution of y will not be known and a deterministic equation for likelihood cannot be derived. In this case, you have to resort to other techniques to estimate the probability density distribution, such as Bayesian Inference or Markov Chain Monte Carlo sampling. In general, you would never have to worry about this in an ML interview, but I'm writing this down because I have a project that does just this and I need to study it to explain it properly.

Solving for epsilon in the original equation and plugging into our distribution, we now can formiulate the probability of y given our input x and parameterized by theta (not conditioned on, because theta is not a random variable).

![](https://latex.codecogs.com/gif.latex?p(y&space;|&space;x;&space;\theta)&space;=\frac{1}{\sqrt{2\pi}\sigma}\exp{(-\frac{(y-\theta^Tx)^2}{2\sigma^2})})

We want to maximize this quantity because we want the most likely values of y and the parameters that achieved that given our input data. This probability is our likelihood, which is explained in the statistics guide. Now, because epsilon was IID, the y's are also IIDs. So the likelihood over the entire dataset can be written as a product of the probabilities.

![](https://latex.codecogs.com/gif.latex?L(\theta)&space;=&space;\prod^n_{i=1}p(y&space;|&space;x;&space;\theta)&space;=\prod^n_{i=1}\frac{1}{\sqrt{2\pi}\sigma}\exp{(-\frac{(y-\theta^Tx)^2}{2\sigma^2})})

To maximize this, we can also maximize the log of this since log is a monotonically increasing function, and also simplifies the calculations. Apply log to the product and simplify using algebra, and you get:

![](https://latex.codecogs.com/gif.latex?l(\theta)=\log&space;L(\theta)&space;=&space;n\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\frac{1}{2}\sum^n_{i=1}(y-\theta^Tx)^2)
![](https://latex.codecogs.com/gif.latex?\frac{1}{2}\sum^n_{i=1}(y-\theta^Tx)^2)

The negative sign was removed to switch from maximizing to minimizing this function. Note that we don't care about anything not related to theta, so we can simplify. We keep the 1/2 because it makes calculating gradients easy. We've derived from scratch the famous least-squares cost function, or our Mean Square Error, and this is what we can perform gradient ascent/descent on.

![](https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha(y-h(x))x)

To avoid overfitting, lasso, ridge, or elastic net regularization is used, though ridge tends to outperform lasso. To add non-linearity, you can add polynomial and interaction terms.

You can also see the CS 229 Stanford Machine Learning course notes [here](http://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) for more details. There is also [this comprehensive article](http://allenkunle.me/deriving-ml-cost-functions-part1) on this topic. 

### Generative models
Models that learn the distribution of the input data instead of the mapping from input to output. In other words, they model p(x|y). These are all classifiers. More details on the below models in the Stanford course [notes](http://cs229.stanford.edu/notes2021fall/cs229-notes2.pdf)

#### Gaussian/Linear discriminant analysis
If the input features are modeled as continuous-valued random variables, and assumed to be multivariate Gaussian, then we can create a generative model that models p(x|y). Assuming y is only 0 or 1, we can model y as a Bernoulli, x|y = 0 and x|y = 1 as separate multivariate Gaussians (the covariance matrices are the same (capital sigma), but the means (mu 0 and mu 1) may be different). The likelihood of the joint probability p(x,y) = p(x|y)p(y) can now be calculated, and the label p(y|x) is calculated via Bayes Theorem. You can then use MLE to find the parameters. 

GDA is very similar to logistic regression, but makes stronger assumptions on the data (the input is multivariate Gaussian). This makes it a stronger fit if the assumptions hold true, but if they don't hold true then the model does poorly. Logistic regression is thus more generalizable if you're unsure of the distribution of the input data.

GDA is actually a general term for both linear discriminant analysis and quadratic discriminant analysis. In QDA, the covariance matrix is not assumed to be the same across classes. GDA/LDA can be commonly used as dimensionality reduction techniques as well. It is similar to PCA in that it tries to project the data onto different axes, but LDA maximizes class separability when projecting the data instead of maximizing variance.

#### Naive Bayes
Naive Bayes also models p(x|y), but makes a strong assumption to make the estimation of p(x|y) much easier. If x was a multivariate random variable, then you would have to model p(x|y) as such, like in GDA. But if x was very high dimensional, this would be unfeasible to estimate. The assumption naive Bayes makes is that every feature in x is **conditionally** independent. This does not mean every feature of x is independent, but rather they are independent **given** the class label. Now instead of estimating p(x1, x2, ... x5000 | y), you can estimate p(x1|y), p(x2|y), etc., which is much easier to calculate log likelihood of. Similar to GDA, the likelihood of the joint probability p(x,y) = p(x1|y)p(x2|y)...p(xn|y)p(y) can be maximized, and then you make a prediction by calculating p(y|x) via Bayes Theorem.

![](https://latex.codecogs.com/gif.latex?p(y|x)&space;=&space;\frac{p(x|y)p(y)}{p(x)})

Because of the assumption, naive Bayes works great on discrete, high dimensional datasets, even in cases when the assumption is clearly not true. It is also very simple, lightweight, and fast to predict. Because of this, it is often used as a baseline for more complex classifiers. It is also a high bias model.

Spam detection is the classic example of using naive Bayes, where x is a one-hot encoded vector of the words out of a fixed vocabulary that is present in an email. 

### Discriminative models
Models that learn to draw a boundary between class labels. Alternatively, models that learn the distribution p(y|x), directly learning the mapping from inputs to outputs. These are also classifiers.

#### Logistic regression
We can follow the same intuition as in linear regression, except instead of y being simply the product of x and the weights theta plus an error term, we want to constrain the values of y between 0 and 1. So, we can apply the sigmoid activation function:

![](https://latex.codecogs.com/gif.latex?h(x)=g(\theta^Tx)=\frac{1}{1&plus;e^{-\theta^Tx}})

The log likelihood of this becomes easy to calculate. y can only take 0 or 1 in binary classification. The probability of y = 1 is simply the above equation. The probability of y = 0 is 1 minus the above. If we combined this into one equation, we get the likelihood of y, and apply the IID trick to get a product of probabilities.

![](https://latex.codecogs.com/gif.latex?L(\theta)=\prod^n_{i=1}h(x)^y(1-h(x))^{1-y})
![](https://latex.codecogs.com/gif.latex?l(\theta)=\log&space;L(\theta)=\sum^n_{i=1}y\log&space;h(x)&plus;(1-y)\log&space;(1-h(x)))

This is what we can perform gradient descent/ascent on. 

![](https://latex.codecogs.com/gif.latex?\theta&space;\leftarrow&space;\theta&space;&plus;&space;\alpha(y-h(x))x)

It's the same update as linear regression. Why that's the case is explained in the Stanford notes, and is outside the scope of this guide.

This is the famous binary cross entropy loss. We can derive the same loss function directly from entropy as well.

**Entropy** - Recall that entropy captures the "surprise" of a random variable, and thus its information content. If we expect certain values from the random variable, then it is not surprising and has little information content. The cross entropy between two distributions is a known formula in statistics:

![](https://latex.codecogs.com/gif.latex?H(p,q)&space;=&space;-\sum^X_xp(x)\log&space;q(x))

p is our target distribution (the ground truth labels) and q is our predicted distribution (h(x)). The ground truth assigns 100% probability to 1 (positive label) or 0% probability to 1 (negative label). So we can write out the sum as the binary cross entropy loss function above, which is just the cross entropy formula expanded.

#### Support vector machines
Support vector machines are supervised learning models that attempt to find a hyperplane that separates two classes. In the vanilla SVM, it finds a linear hyperplane that maximizes the margins between the data points, so it doesn't just find any line that can separate the data points but one with the best distance between points that are close to the boundary and are difficult to classify. They are very similar to logistic regression models, the primary difference is the hinge loss instead of logistic loss and the kernel trick. Additionally, labels are {-1, 1} instead of {0, 1}. SVMs are considered some of the best off-the-shelf supervised learning classifiers.

**Hinge loss** - Intuitively, we need a loss function that provides 0 loss for points that are outside their correct margin and starts penalizing them more strongly the further the are on the wrong side of the margin. The natural function that could represent this is the ReLU function, which is flat until a central point (at the margin), and then becomes linear slope. This is exactly what hinge loss looks like.

https://latex.codecogs.com/gif.latex?l=max(0,&space;1&space;-&space;y(w^Tx&space;&plus;&space;b))

You can see that this returns 0 when the model correctly predicts -1 or 1 or even higher magnitude in the correct direction, and linearly increases the further away the model's prediction is from the correct label. 

**Kernels** - Let's say you're performing linear regression but the model is suffering from high bias and is not fitting well. You could decide to use higher-order features and some interactions to increase the variance of the model, say squaring each term and adding interaction terms. We'll define this as a mapping &phi;(x) that projects the original input space to the new higher-dimensional space that the model will use. If your input space already has a high dimensionality, adding these terms will significantly increase run time and complexity of the model. Can we taken advantage of these higher-order features without sacrificing run time? Kernels allow you to do this via the kernel trick. A kernel is, mathematically speaking, the dot product between the mappings of two vectors.

https://latex.codecogs.com/gif.latex?K(x,z)=\langle\phi(x),\phi(z)\rangle

Okay, but don't you still have to compute the mapping to compute the dot product? Why are we computing dot product? That's because when we replace x with &phi;(x) into the gradient update rule with least mean square (see [linear regression](#linear-regression)) and simplify we end up with a dot product between a sample and all other samples. The second part is the _kernel trick_, which is when a kernel function, or the inner product of some mapping, ends up magically cancelling out terms or simplifying to where you can compute the dot product in O(original input dimension) time, effectively negating the cost of added complexity. I can't really explain it any other way without showing you some example, and for that I will defer you to the Stanford [notes](http://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf). Now we can exploit the higher-dimensional space without ever having to actually calculate those projections ourselves. So the kernel trick let's us add more complexity to the model without significantly increasing computational cost or having to add more features and collect more data.

The kernel trick enables the flexibility of SVMs (or any discriminative model) to handle classes that are not necessarily linearly separable. Perhaps the data is separable in a higher dimension, or some complex mapping from the current space to a higher-dimensional space. Radial basis function, or RBF, is a popular kernel that is used in SVMs that essentially uses the distance from the origin or some point to the input vectors as the new embedding. 

**SVM vs Logistic Regression** - Hinge loss penalizes any sample that is either within the margin or outside the wrong margin linearly scaling with how far it is. It also requires that classes be 1 or -1 instead of the usual 0 and 1. But, once the class is outside the correct margin, there is no loss associated with it. This is unlike logistic loss, where there is a probabilistic interpretation of how likely a sample belongs to a class. So you can be above 0.5 probability for a class but still incur large errors if it's far from 1. This enables logistic loss the flexibility of having probabilistic interpretations, in cases where you want to know how likely a sample belongs in one class or multiple. Alternatively, SVM's hinge loss does not penalize samples once they are correctly classified. So, it does not have a probabilistic interpretation, but this tends to increase accuracy. SVMs do not fare well when there is overlap between classes and no clear margin can be found.

#### Perceptron/Linear classifier
This is essentially logistic regression, except uses a step function instead of sigmoid. So instead of a fuzzy label from 0 to 1 to predict probability of a postive label, you have a hard cutoff for either 0 or 1. Essentially, if wx + b > 0, predict 1, otherwise predict 0.

### Decision trees/CART
Decision trees or CART (classification and regression trees) are a non-parametric supervised learning algorithm that make multiple splits on a dataset to best separate all the samples into all the classification categories. It does so in a greedy manner at each decision to minimize heterogeneity. For classification, this can be information gain (a decrease in entropy) or minimizing Gini impurity, both of which are measures of heterogeneity. For regression, typically MSE from the mean of the split subset is minimized. Effectively, decision trees is a nonlinear model by constructing multiple linear boundaries.

Stopping criteria for decision trees are very important for controlling overfitting. There are many criteria that can be used. You can:
- Set a max tree depth
- Set minimum number of samples in a leaf/terminal node
- Set a max number of terminal nodes
- Set a minimum number of samples to split a node
- Set a minimum decrease in impurity or increase in information gain. If splitting does not produce more homogenous groups, then stop splitting

The advantages of decision trees is that they’re very easy to interpret, are good for feature selection by finding features that were most important for the splits, and require minimal preprocessing for the data. They can also handle categorical  and multiple classes really well.

The disadvantages are that they overfit very  and are very sensitive to new data points or outliers. There are many methods to address this however. For example, pruning is used to remove leaf nodes that contribute little to information gain or classification accuracy. Random forests is an ensemble method that uses bagging with multiple decision trees to prevent overfitting. Actually, typically random forests or gradient boosted trees are generally used over individual decision trees. Decision trees also do not do well with imbalanced data, they tend to underfit.

**Gini impurity** – measure of how likely it is that a sample will be classified incorrectly. Higher means more heterogenous with a max of 1.0, lower means more homogenous with a min of 0. It is calculated by:

![](https://latex.codecogs.com/gif.latex?G&space;=&space;\sum_i^Cp(i)(1-p(i)))

So if we have categories that are evenly probable in a dataset, Gini impurity would be maximal. For binary classification, that would be 0.5 * 0.5 + 0.5 * 0.5 = 0.5. With more categories, this increases and approaches a value of 1.

**Entropy** – measure of uncertainty, or heterogeneity as used in decision trees. A high value indicates higher heterogeneity. Information gain is a decrease in entropy. Entropy involves log calculations, so gini impurity is typically used more.

These are some great resources on decision trees: 
- https://www.youtube.com/watch?v=_L39rN6gz7Y&t=0s
- https://towardsdatascience.com/decision-trees-d07e0f420175

### Ensemble methods
#### Random Forest

Random Forest is an ensemble of decision trees that can be used for classification or regression by aggregating the outputs of all the individual trees. There are two main features of Random Forests that make it more powerful than individual decision trees: 
1) it uses bagging (bootstrapping aggregation), where the training dataset is resampled and each tree is trained on a different resample/bootstrap
2) trees make decisions on a subset of features instead of all. 
Both of these features allow the individual trees to become more uncorrelated opinions, allowing them to cover the errors of the other and improving accuracy, reducing overfitting.

The most important hyperparameters for random forests are the number of trees and the number of features to use for each tree. Increasing the former usually improves performance upto a certain point, but increases computation time. This also decreases variance. Increasing the latter increases variance and reduces bias, decreasing it will reduce variance and increase bias.

Increasing depth of each tree increases variance.

Increasing the minimum number of samples at each leaf node forces a regularization effect because it prevents the tree from becoming too complex. It increases bias.

https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

#### Boosting

While bagging randomly samples a dataset and trains many learners in parallel, hoping to reduce variance of each learner by combining the votes of many uncorrelated learners, boosting reduces bias by sequentially building new learners on the mistakes of past learners. Thus, boosted trees are more prone to overfitting than random forests and are also more difficult to tune, but generally see better performance. Additionally, random forests have slower run time due to the high number of trees, and gradient boosted trees have much faster run time, which is why they're often use for search ranking. Random forests are generally more effective for datasets with a lot of noise. Boosted tree ensemble methods usually perform better on datasets with less noise and handle imbalanced datasets well. A good link explaining the differences can be found [here](https://www.datasciencecentral.com/profiles/blogs/decision-tree-vs-random-forest-vs-boosted-trees-explained). The StatQuest [series](https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6) on boosting is also quite helpful. 

**Adaboost** - sequentially builds on weak learners (almost always stumps, or just one split) by changing the weights of samples - more weight for misclassified and less weight for correctly classified - and resampling according to those weights to fit new learners. Thus, new learners build on the mistakes of the previous. Trees contribution to final decision are weighted by how well they performed.

**Gradient boost/MART** - Gradient Boosted Trees or MART (Multiple Additive Regression Trees) if they are applied for regression, does the same except by fitting trees to the pseudo residuals (calculated by the gradient of the loss function) of the last tree, and using a learning rate to weigh every tree's contribution to the final decision. The number of trees you use can cause overfitting, unlike in random forest where increasing number of trees does not induce overfitting. Unlike AdaBoost, MART does not use stumps and trees have depth. Classification is very similar to regression, except with cross entropy loss and applying sigmoid to convert continuous value output of leaves to a probability. Steps for fitting a MART:
1) Start with an initial prediction, a singular value that minimizes the loss to each predicted value
2) Compute the pseudo residuals by taking derivative of loss function. If loss function is MSE, then derivative is just the residual, but different loss functions can be used, hence "pseudo" residual.
3) Fit a new tree to the residuals. Splits are determined by variance gain, which uses the gradients of the loss function. The formula derived in the original [paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf). Tree fitting stops with maximum depth, or maximum number of leaves.
4) Find the new singular value that minimizes the loss at each leaf (by calculating gradient of loss function).
5) Add the leaf values to previous predictions multiplied by a learning rate
6) Repeat 2 - 5. Stop when a certain number of trees is reached, or loss stops improving more than a certain amount.

**XGBoost** - Extreme gradient boosting is a specific implementation of gradient boost that adds a few key features that make it significantly more powerful than vanilla gradient boost. These are regularization, parallelization, and the second derivative of the loss function. 
1) XGBoost uses a modified objective function. Instead of using the loss function to compute the residuals on leaf nodes or variance gain at splits, we used a second-order Taylor series expansion on the loss function plus a regularization function. This new objective function is used to determine the score gain for a split and the new residual on a leaf. The advantage of using the Taylor series expansion is that it allows XGBoost to be used with any loss function, differentiable or not. A great example is pairwise ranking. 
2) The regularization is also unique for tree models, as it penalizes complexity and addresses the common problem of overfitting for boosted tree ensemble models. It is proportional to the number of leaves and the scores on the leaves. The regularization term is included when calculating gain or residuals.
3) It parallelizes WITHIN a tree but not multiple trees, since they must be fit sequentially. This is more of an implementation detail of the package that XGBoost was released in, rather than an algorithmic detail.
I point you towards the official [documentation](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) for XGBoost for more info on the math.

In a way, boosted tree ensembles are similar to neural networks in the sense that they use gradients of the loss function to improve their nonlinear decision boundaries. However, unlike neural networks there are no "parameters" to update in tree models (they are non-parametric), just where the splits occur. The advantage of boosted tree ensembles over neural networks is that they are faster to run, train, require less data, and can handle mixed feature types. Neural networks are still capable of learning complex functions if given enough data, moreso than tree models. So tree ensembles are great if you need more complexity than the regression / SVM models, but you are too constrained on data or runtime for a neural network. Of course, more complex tasks such as vision and language tasks are entirely in the realm of neural networks.

### Neural networks
Neural networks are a field of their own (see [Deep Learning](https://github.com/RdoubleA/MLprep/blob/master/deep_learning.md)), but the most simple neural network, the multilayer perceptron, is used in the same league as traditional machine learning algorithms.

Think of a multilayer perceptron as multiple stacked linear regressions, but in between you inject a nonlinear activation function such as sigmoid, ReLU. This gives neural networks the advantage of approximating almost any function, allowing it to outperform nearly every other machine learning algorithm, but at a cost of course. These are really only used for very complex mappings, such as voice recognition or face detection.

Advantages:
- superior performance to other ML models

Disadvantages:
- requires a VERY LARGE amount of data
- black box, not interpretable at all
- computationally costly to train
- slow run time

### Optimizers
See the [Deep Learning guide](https://github.com/RdoubleA/MLprep/blob/master/deep_learning.md).

### Model evaluation
For evaluating a linear regression model, see the statistics guide. For evaluating a model in deployment, see the system design guide. 

Generally, evaluating your model is simply the loss function and its performance on training/validation to assess overfitting. For classifiers, there are many metrics to be aware of.


#### Classification Metrics
**Accuracy** - Accuracy is the total correct divided by all total classifications. Or, (TP + TN) / (TP + FP + FN + TN). Useful only when classes are balanced. If imbalanced, accuracy is not meaningful, i.e., detecting cancer that happens at a 1% rate in a population. You can just say no all the time and get a 99% accuracy.

**Precision** - Precision measures the proportion of predicted positives that were truly positive. That is, TP / (TP + FP). In the cancer example, we never predicted a true positive, thus precision is 0. Precision is useful when we want to be very sure of our prediction. For example, we don’t want to falsely diagnose someone with cancer, so we optimize for precision. Increasing precision decreases our chances for Type I error, or when we wrongly predict positive for an actual negative result. However, since we’re taking extra care to be precise, we may let a lot of cases that actually have cancer slip by undetected, which could also be problematic. In other words, precision will minimize false positives, but may not minimize false negatives. Chance of Type I error decreases but chance of Type II error will increase. Depending on your problem, this may be ok.

**Recall** - Recall measures how many actual positives were correctly classified. It is useful when we want to make sure to capture all the positives. For example, if we are trying to predict terror threats, we want to capture all potential threats even if it may be a false alarm. Maximizing recall minimizes chance for Type II error, or missing a threat that was actually there. However, if you only maximize recall, you can get a recall of 1 if you predict 1 for every example. Thus, recall will minimize false negatives but may inflate false positives. Chance of Type II error will decrease, but chance of Type I error will increase. 

**F1 Score** - Precision and recall are two opposing forces. You want to be sure when you predict cases with cancer, but at the same time you want to be able to identify all cancer cases as much as possible, thus the tradeoff between precision and recall. Precision vs recall is the same tradeoff of Type I vs Type II error. F1 score measures how well the model manages this tradeoff, as it is a harmonic mean between the two. You can also use domain knowledge if you want to weigh finding all the positive cases more or being more confident in your positive classification more by the F beta score, which adds a factor to change the balance between the two. Increasing beta increases how much we care about recall, or minimizing false negative.

https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226

**Sensitivity** - Sensitivity is the true positive rate, or how many of the actual positives were correctly classified. It is equivalent to recall, statistical power, and the inverse of Type II error. 

**Specificity** - Specificity is the true negative rate, or how many of the actual negatives were correctly classified. It is like precision for negatives.

**ROC AUC** - Both specificity and sensitivity come into play when using AUC ROC as a classification metric. When you vary the probability threshold used for classifying an example as positive, you change the model’s sensitivity and specificity. Increase the threshold (in the range of 0.5 - 1.0), and you get fewer positives, which will decrease sensitivity, FPR, recall and increase specificity, precision. Decrease the threshold and you will get more positives, which will increase sensitivity, FPR, recall and decrease specificity, precision. This is plotted by an ROC curve, which plots sensitivity against FPR, or 1 – sensitivity, or the fraction of negatives that are incorrectly classified, over various thresholds. The area under this curve measures a model’s ability to separate classes. An AUC close to 1 indicates that the model is able to perfectly distinguish classes. An AUC close to 0.5 means the model cannot distinguish classes at all. An AUC of 0 means the model is completely reversing the classes. 

In multi-class problems, you compute ROC curves for one class vs all for each class. 

AUC is a great metric for general performance of a model irrespective of the threshold used. However, when the costs for false negatives or false positives are imbalanced, or the classes are heavily imbalanced, then AUC is not that informative. AUC works wells when there's less class imbalance and you care equally about false positives and false negatives. When working with many negatives and few positives, false positive rate will always be very low and will inflate the AUC score. In a way it's a better form of accuracy. F1 score is more appropriate for imbalanced cases, or you could use area under precision and recall curve. F1 score is especially good if you care about the positive class, since it is an average of measures of how well you classify the positive class, precision and recall. F1 score is calculated at a probability threshold, whereas AUC is calculated across thresholds.

https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

### Loss functions
#### MSE vs MAE

L1/MAE loss is not fully differentiable, so training may be less stable, but it is less sensitive to outliers since it gives equal weight to large error and small error. L2/MSE penalizes outliers more heavily, thus the model ends up becoming more sensitive to outliers. But it is fully differentiable and thus training is more stable. Both are generally pretty robust.

## Unsupervised Learning
Models that identify patterns in data without any training labels. They don't necessarily learn a mapping from input to output.

- [Clustering](#clustering)
  * [K-means clustering](#k-means-clustering)
  * [Mean-shift clustering](#mean-shift-clustering)
  * [DBSCAN](#dbscan)
  * [Gaussian mixture model with EM](#gaussian-mixture-model-with-em)
  * [Agglomerative hierarchical clustering](#agglomerative-hierarchical-clustering)
- [Dimensionality reduction](#dimensionality-reduction)
  * [ICA](#ica)
  * [PCA](#pca)
  * [t-SNE](#t-sne)
  * [UMAP](#umap)

### Clustering
#### K-means clustering

Initialize a predetermined number of centers. Calculate cluster membership for each data point by computing distances to every center and picking the closest one. Recalculate new centers by finding the mean data points for each cluster. Repeat.

The [general approach](https://www.codecademy.com/learn/intprep-ds-machine-learning-algorithms-ds-interviews/modules/intprep-ds-unsupervised-learning-interview-questions/cheatsheet) for determining the optimal K is by calculating the _inertia_ or the sum square of the residuals/distance between every data point and its center and summing them up for every value of K. Inertia will decrease as you increase K, and there is usually an "elbow" where inertia starts to decrease more slowly. This point would be a good number of clusters.

Advantages: Runs in linear time complexity, O(n), so is most scalable
Disadvantages: Need to determine number of cluster beforehand, different results every run due to random initialization, restricted to point cloud clusters and not arbitrary shapes

#### Mean-shift clustering

Initialize a sliding window with radius r and randomly initialize its center. Compute the mean of all the points within the sliding window and move the window to the new center point. Repeat until the number of points inside window does not increase. Repeat this procedure for additional sliding windows until all points are in a window. Remove overlapping windows at the end and keep the one with most points, and identify clusters by assigning points to closest window center.

Advantages: No need to select number of clusters, automatically discovers this
Disadvantages: Need to select radius r, 

#### DBSCAN

Choose a random point, a neighborhood distance epsilon, and a minimum number of points. If there is at least the minimum number of points within the neighborhood, begin clustering. Mark this point as visited, assign all points in the neighborhood to this cluster, and go to next unvisited point within this cluster. The neighborhood will keep expanding as you process more points. Continue until all points within this cluster is visited. Go to next unvisited point after this cluster is complete. If not enough points to start cluster, then label point as noise.

Advantages: works with arbitrary shapes of clusters, identifies with outliers, do not need to predefine number of clusters
Disadvantages: does not work well when clusters have different densities, high dimensionality when neighborhood distance epsilon is difficult to estimate

#### Gaussian mixture model with EM

Choose a number of clusters and randomly initialize parameters for a normal distribution for each cluster (mean and std). Compute each point's probabilities for being in each cluster. Use these probabilities to update distribution parameters using EM. Continue until convergence.

Advantages: unlike K-means, can detect elliptical shaped clusters. supports mixed membership
Disadvantages: does not work as well with arbitrary shapes of clusters

#### Agglomerative hierarchical clustering

Initialize each data point as its own cluster. Compute average linkage between each pair of clusters, or average distance from points in one cluster to another. Combine two clusters with the smallest average linkage. Repeat until all data points in one cluster.

Advantages: detects hierarchical structure, allows freedom to choose number of clusters, works well with any distance metric
Disadvantages: O(n^3) time complexity

Here is a great [resource](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68) explaining the above five. [sklearn](https://scikit-learn.org/stable/modules/clustering.html) also has a great graph depicting the differences between all the clustering methods. 

### Dimensionality reduction
#### ICA
Independent Components Analysis tries to find an alternative subspace to project the data onto such that the new vectors (sources) are all independent and it's a linear combination of the original vectors. It makes three assumptions: that the sources are independent and non-Gaussian and they are mixed linearly. The algorithm itself does this by a MLE method of the unmixing matrix, the [details](cs229.stanford.edu/notes2021fall/cs229-notes11.pdf) are out of scope of this guide. A famous example is the cocktail party problem, where you have an audio recording at a cocktail party and you are trying to separate the independent voices. 

#### PCA
Principal Components Analysis is similar to ICA in that it is trying to find a new projection of the data. However, the goal is very different. Instead of independence, it wants to maximize the variance captured in the new vectors (more on difference between ICA and PCA [here](https://www.quora.com/What-is-the-difference-between-PCA-and-ICA)). To do this, PCA is simply the SVD of the covariance matrix. The eigenvectors serve as the new subspace to project your data on. PCA is used most often for dimensionality reduction by taking the largest n eigenvectors. You can use the same method as in k-nearest neighbors, where you keep adding the components with the largest eigenvalues until the cumulative variance captured stops increasing at the elbow point, or past a certain amount.

#### t-SNE
t-distributed stochastic neighborhood embedding is a nonlinear dimensionality reduction technique, unlike PCA and ICA. It essentially uses the idea that points that are close together in the higher dimensionality should be close together in the lower dimensionality. Determining what points are "close together" is done by modeling the distances between points as conditional probabilities using Gaussians, basically saying what is the probability that point A will choose point B as its neighbor. The new lower dimensional space will have the point pairs modeled with the t-distribution (the reasons for which are in the original [paper](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)). Then, the algorithm simply minimizes the KL divergence between these two distributions via gradient descent.

t-SNE is generally run for visualizations, especially for seeing different classes separate on the t-SNE dimensions. Typically, you run PCA first and then t-SNE on a smaller number of features since it is computationally expensive. This is one drawback. Another is that you have to tune the perplexity parameter, which essentially affects the neighborhood size of the points. A third is that it's stochastic and you won't always get the same results. But t-SNE does work well with data that aren't linearly separable or have arbitrary shapes.

[This](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a) article goes into more detail.

#### UMAP
Uniform Manifold Approximation and Projection is a recently published nonlinear dimensionality reduction technique that claims several advantages over t-SNE, namely better preserving global structure and faster run time. It is similar to t-SNE in that it tries to maximize the similarity between point neighborhoods in the lower dimensions and in the higher dimensions. It does this by creating a fuzzy graph, where points are connected by an edge if they are their n-th nearest neighbor, which is a parameter set by the user. All other points have a fuzzy edge with a probability based on how far you have to expand the radius out. UMAP then tries to create the most similar graph in the lower dimensional space.

More on UMAP [here](https://pair-code.github.io/understanding-umap/).

## ML Concepts

- [Class imbalance](#class-imbalance)
- [Training, validation, test sets](#training--validation--test-sets)
  * [Cross Validation](#cross-validation)
  * [What if the distribution of test data is different from distribution of training data?](#what-if-the-distribution-of-test-data-is-different-from-distribution-of-training-data-)
- [Data imputation](#data-imputation)
- [Hyperparameter search](#hyperparameter-search)
- [Regularization](#regularization)
  * [Lasso vs Ridge regularization vs Elastic Net](#lasso-vs-ridge-regularization-vs-elastic-net)
- [Feature importance](#feature-importance)
- [Feature selection](#feature-selection)
- [Bias/Variance tradeoff, overfitting, underfitting](#bias-variance-tradeoff--overfitting--underfitting)
- [Generative vs. discriminative models](#generative-vs-discriminative-models)
- [Parametric vs. non-parametric models](#parametric-vs-non-parametric-models)
- [The Curse of Dimensionality](#the-curse-of-dimensionality)
- [Outliers](#outliers)
- [Similarity metrics](#similarity-metrics)
- [Feature engineering](#feature-engineering)

### Class imbalance
When classes are heavily skewed (90% yes, 10% no), the model can have trouble learning the minority class and predicting future minority class examples. To address this, you can:
 - Undersample the majority class and/or oversample the minority class (bootstrapping, repetition, etc)
 -  Weigh the minority class more in the cost function
 -  Use a more appropriate accuracy metric, such as F1 score

### Training, validation, test sets

#### Cross Validation

Cross validation is a validation method that partitions the dataset into folds, where all but one fold is used for training and the last fold is used for validation. This validation fold is rotated among all the folds and the model’s validation accuracy is averaged across all these experiments. Cross-validation is preferred over normal hold out validation because you are able to use all of the data to inform the model, and you can better evaluate the model's bias and variance. A model with high variance will have high standard deviation of validation errors across all folds, a model with high bias will have high mean validation error across all folds.

There are multiple forms of cross validation. Leave one out cross validation trains a model on the entire training dataset and validates on one example. K-fold cross validation splits dataset into k folds and trains a model on k-1 folds and validates on the last, then this process is repeated and the validation fold is rotated. A very large k becomes leave one out CV, and a very small k would mean shrinking the training size. A small k will decrease testing variance because there is a large set of data to validate on and the error estimate will vary less due to noise or outliers. However, our model trains on fewer data points, so estimates of model performance may be biased. On the flip side, a large k will increasing testing variance because you are validating on fewer points, and error estimates will vary a lot due to noise and outliers. However, our model trains on more data points, so estimates of model performance will be less biased.

This [link](https://codesachin.wordpress.com/2015/08/30/cross-validation-and-the-bias-variance-tradeoff-for-dummies/) will explain further.


#### What if the distribution of test data is different from distribution of training data?

This will make models harder to perform well on the test data set. It will have high accuracy on the training set but not the test set, meaning it will overfit. You can mitigate this by mixing some samples from the test distribution with the training distribution and making sure the validation set matches the test set distribution as closely as possible, otherwise it's a moving target for the model. Cross-validation with different splitting can help as well.

You might also consider using high bias models that can generalize better.

Differences in distribution may occur when training samples are obtained in a biased way, or maybe a different time period with different circumstances.

### Data imputation
When you have missing data, you can approach it the following ways:
- Replace all values with the mean/median value for that feature. However, this does not take into account the correlations between features and reduces the variance of this feature. It also does not work for categorical variables
- Replace values with the mode. This only works for categorical variables. Again, the issues with mean replacement also apply
- Remove samples with missing values. Only works if you have enough data.
- Train a model to predict the missing value. kNN is commonly used, to predict missing features based on the similarity of other features with other data points. Or a generative model.
- Repeat samples from the rest of the dataset

### Hyperparameter search
1) Grid search - select some values for each hyperparameter and try every combination, like a grid
2) Random search - select a range of values for each hyperparameter and randomly choose values

Random allows you to try more different values in case one hyperparameter does not improve objective function at all - in which case grid search will be redundant for different values of the dud hyperparameter.

Some parameters you should sample on a log scale instead of uniformly at random, especially learning rate which can vary on a log scale (0.0001 to 1). For example, sample the exponent uniformly at random and then use learning rate = 10 ^ that exponent. OR for values that range from 0.9 to 0.999, use 1 - 10 ^ that exponent

Bayesian optimization uses performance of previous searches to find the next best one, and tends to perform better than random, grid, or manual search. You can implement in Python using hyperopt.

Andrew Ng's [lecture](https://www.youtube.com/watch?v=AXDByU3D1hA
) on this topic.

### Regularization
Regularization is a set of techniques to prevent a model from overfitting to the data by penalizing model complexity. It usually involves adding an L2 or L1 norm of the weights term to the loss function. Using the L1 norm is lasso reg. L2 is ridge reg.

#### Lasso vs Ridge regularization vs Elastic Net

Regularization shrinks the coefficients of a model and reduces overfitting. Lasso regularization/L1 enforces sparsity, as it is capable of driving weights to zero. Thus, it may even be used for feature selection. Good for when you know not all features will correlate with prediction. Ridge regularization/L2 ensures that no weights go to zero, but it is not robust to outliers unlike L1. It is good when all the features are important for the output. Elastic net uses both a linear and square term of the weights in the loss function and combines the advantages of both ridge and lasso.

### Feature importance
In linear models, you can use the coefficients to compute feature importance, or p-values from linear regression. In tree-based models, feature importance is calculated by mean decrease in Gini Impurity or whatever metric the model used to determine the split for that feature (this is used in sklearn). You can also use sum of number of splits including that feature weighted by the number of samples it splits. One drawback of MDI is it assigns higher importance to high cardinality features, especially continuous variables, so other methods such as permutation importance are also used. (scramble the values in that feature column and compute decrease in impurity, repeat many times and calculate average).

### Feature selection
There are several ways to remove features:
1. Ablation trains a model on all the features then removes one at a time and see the decrease in model performance to determine importance
2. L1 regularization tends to send coefficients of unimportant features to zero, and can be used for feature selection
3. You can use simple statistical measures, such as variance of the feature to remove low variance features. Another measure is p-value of statistical tests with output values, or even correlation values.
The sklearn [page](https://scikit-learn.org/stable/modules/feature_selection.html) on feature selection is quite useful.

### Bias/Variance tradeoff, overfitting, underfitting
Model error due to bias is a result of underfitting, or that the model is too simple and makes too many prior assumptions about the data to accurately fit it. Examples of high bias models are linear regression, since it assumes the data is linear, for example. If we increase model complexity we can find more complex patterns, but that leans towards overfitting the data and causes the model to generalize poorly. This is error due to variance, meaning the model is too complex and is fitting noise in the data instead of the general pattern. You can think of it as the model varies highly when changes datasets and generalizes poorly.

Thus, whether to increase model complexity or decrease model complexity is a tradeoff between bias and variance. High bias means better generalization but poorer fit, but high variance means excellent fit but poor generalization. The ideal model is somewhere in the middle.

You can determine whether a model has high bias or variance by examining the training and validation/test errors. A larger validation error than the training error indicates the model generalizes to unseen data poorly and is overfitting the training data, therefore it has high variance. If both training and validation error are high, then the model is underfitting the data and has high bias. There are many methods to address these issues:

Underfitting – expand the model capacity (add more trees, hidden layers, dimensionality, etc), add more input features, remove regularization

Overfitting – use regularization, decrease model capacity, use dropout (in neural networks), get more training data (helps model distinguish signal from noise and generalize to new data better), use ensemble methods, remove input features, early stopping

### Generative vs. discriminative models
Discriminative models try to parametrize the posterior distribution p(y | x). That is, given a distribution of data, what is the most likely classification. It does not care about how the data was generated or what the distribution looks like. Alternatively, generative models attempt to parametrize p(x | y). That is, given samples of a data distribution, what is the underlying distribution that produces these. For example, a generative model shown images of cats would attempt to generate images that look similar to those cats, but discriminative models would try to classify all cat images as cats.

### Parametric vs. non-parametric models
Parametric models are defined by a constrained number of parameters that do not scale up with the size of the training set. On the other hand, nonparametric models increase number of parameters with more training samples. For example, linear regression, logistic regression, and SVMs are parametric models but decision trees, KNN are non-parametric models

### The Curse of Dimensionality
When data dimensionality increases, data points become more equidistant from each other. This make its significantly more difficult to extract any meaningful structure from the data, especially for clustering methods. Additionally, searching the solution space with greater dimensions becomes computationally more complex and your model is prone to overfitting. More dimensions means exponentially more space, meaning more space the model has to generalize to!

### Outliers
L1/lasso regularization can increase bias of the model and is also resistant to outliers. L1/MAE loss is more resistant to outliers than L2/MSE. Tree-based methods also tend to be more resistant than regression methods.

You can also transform the data (i.e., BoxCox transform) or remove outliers or Winsorize (clip) the outliers.

### Similarity metrics
Most commonly you will see Euclidean and Cosine
1) Euclidean distance - sqrt of sum of squares differences between points. Pretty much pythagorean theorem. Also known as L2 norm.
2) Manhattan distance - sum of absolute differences between points. Like traveling in a street grid in Manhattan. Also known as L1 norm.
3) Minkowski distance - generalization of the above two, where the differences between points are taken to the nth power, summed, then nth-rooted. When n = 1, you have Manhattan distance, 2 is Euclidean, infinity is Chebyshev.
4) Cosine similarity - the angle between the two points as vectors. Take the dot product between the two points and divide by the L2 norm of both points. If close to 1, then angle is 0 degrees and vectors are near identical, positively correlated. If close to 0, then they are orthogonal and not correlated a bit. If close to -1, they are opposite and negatively correlated.
5) Jaccard similarity - this only works on sets. It is essentially the number of elements the two sets share (double counted since there's a copy in each set) divided by the total number of unique elements in both sets, or intersection divided by union

### Feature engineering
Feature engineering involves designing metrics that transmute actions and qualities of a user or item into something usable by a model. For example, measuring how much time a user spent watching a video to capture user engagement. In the specific scenario you are building the ML model for, it is always important to consider all the actors (user, product, context/historical data) and their interactions (user-product similarity, etc) to come up with features. After you get these metrics, you have to clean them. Raw data is usually not in the proper format for a machine learning model to train effectively on it. Categorical variables need to be encoded for certain models. Some features have skewed distributions, and you might consider log transforming them so that the model can work with a wider range of values. There may be outliers that you may want to remove or cap / Winsorize. Missing data needs to be either removed or imputed with the mean value, or some other data imputation method. This [article](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114) discusses cleaning features in detail.

## Industry ML

- [Recommendation systems](#recommendation-systems)
  * [Candidate generation](#candidate-generation)
  * [Ranking](#ranking)
- [Learning to rank - ML for search engines](#learning-to-rank---ml-for-search-engines)
  * [RankNET](#ranknet)
  * [LambdaRank](#lambdarank)
  * [LambdaMART](#lambdamart)

### Recommendation systems

Recommendation systems are very important for many common services, such as suggesting movies you may like, or finding a product similar to the one you bought. It does this by gathering user-to-item relationship data and calculating similarities between users or between items to find the best new item for you.

Let's assume you are tasked to build a recommendation system that recommends movies to users on Netflix. I walk through this system design exercise in detail in the ML System Design guide. Here, I will focus on the models.

In brief, we will structure our model in two stages. Because of the scale of the problem (53 million DAU, millions of movies/shows to choose from), we cannot just apply a complex model. We will first select relevant movies from the large corpus of data using a candidate generation model that should be simple and have low run time. This will focus on recall, or getting any relevant items out of the total list of items. Then, we will rank the relevant items based on user preferences using a more complex model. This will focus on precision, or getting the most relevant items listed in the correct order out of all the relevant items. 

#### Candidate generation

**Collaborative filtering** - In this approach we will use data from similar users to the current user to predict whether a movie should be recommended or not. User-to-item relationships can be represented in a matrix. The actual value could be explicit feedback, where the user physically rated an item, for example, or implicit feedback, which uses other metrics that imply the users satisfaction with an item, such as time spent watching a video. These can be represented as vectors for every given user, or for every given item. You can remove user biases by subtracting average user ratings from each rating. Collaborative filtering can use two approaches to predict user rating for a new item. 
1) _Nearest neighborhood_ uses the similarities between user vectors to predict a user's rating for a new item based on the top K users. Cosine similarity is typically used, then take the top K users vectors, select the value corresponding the the new item, and average, weighted by similarity. This is the predicted engagement of this new item for the current user. This process can be computationally expensive when you have a massive user base and movie collection. It is also sparse - there could be new movies that have no user engagement, or the user may be new. 
2) To address the sparsity, another method called _matrix factorization_ is done. The idea is to find a latent space that contains salient features for each user and each item that is hidden from our metrics. These latent vectors might represent the quirkiness or dark humor content of a movie or preferences of a user, for example. Once we have the latent vectors for both, we can compute the dot product of a user's latent vector with an item's latent vector. Essentially, we want to factor a single matrix users x items into two matrices, user x latent and latent x items. If you're familiar with SVD, this might seem familiar, except SVD produces three matrices, including the diagonal matrix of eigenvalues, and it also does not work well with sparse data. Since we have a very sparse matrix, we cannot do this analytically, we must approximate it with a cost function and use gradient descent. So we choose the dimensino of the latent space, randomly initialize the user-latent and latent-item vectors, then use mean square error or absolute to compute the difference between its dot product and the actual rating, then we using alternating least squares / gradient descent to optimize. This textbook [chapter](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf) from Stanford goes through collaborative filtering in detais

**Content-based filtering** - Content-based approaches typically require a lot of inherent information about the content itself independent of the user. You can define a vector for each item where features can be actor, genre, mood, year came out, topic, etc. Then convert these scores (usually binary) to TF-IDF. This is your media-profile matrix, now you can use the vector for each movie to compute dot product with a user-profile matrix. The user-profile matrix can have the same features, and the values can be their implicit feedback for movies with those features. You can simply take the most similar movies to the user and recommend those, or the most similar to movies already watched and recommend those.

**Embedding similarity** - Similar to matrix factorization, we can learn the latent vectors using neural networks instead, which are more flexible with sparse data. The architecture will be two towers - essentially we have one encoder (hidden layers get progressively smaller) for each users and movies. The encoders output an embedding vector. The loss function to optimize is the different between the dot product similarity of these vectors and the actual implicit feedback label of this user to this movie. Once you've trained the network, you don't use this directly for predictions but you can generate the latent vectors for users and movies. Then, you can take the K nearest neighbors in the embedding space as suggested movies for a particular user.

Collaborative filtering and embedding similarity suffer from the cold start problem - for new users or new movies there is no historical data to train with, or find similarity to. Content-based filtering is ideal for scenarios involving new users/content, because you can create profile vectors for a new movie, or ask a user their preferences during onboarding. So maybe you can use this in the beginning and switch to another method after collecting more engagement data of the user. Another advantage of collaborative-filtering is that it does not require domain knowledge to create user vectors since it is purely based on historical engagement. Actually, you can use all three methods to generate scores to be used by the ranker.

#### Ranking
Ranking in recommendation systems is different from search in that the relative ordering in the top recommended is not as important, so this is framed as more of a logistic regression problem than a pairwise regression problem. We want to predict the probability of the user watching the movie given all the scores from the candidate generation stage. So we can start simple with a logistic regression or random forest model if we are limited in training data, model capacity, or just need a baseline. If we have plenty of resources, you could use a 2-layer feedforward NN with ReLU activations and binary cross-entropy loss.

### Learning to rank - ML for search engines
Learning to rank is a different category of machine learning problems. Most models fall under classification or regression where the model has to output a single value or label. In LTR, the model has to output an optimal ordering of a list based on a cost function. In search engines, the goal is to produce an optimal ordering of documents for a given query such that the most relevant documents are listed first. Here, run time is critical so that user's can find the result of their search query in millisecond scale time, so heavy deep learning models to parse the text of documents are out of the question. Since the dataset is massive (billions of websites), there is usually a staged process for making a prediction, where more relevant documents are progressively filtered through simpler models. Here, we describe the final model that makes the final order of documents for a query. The most common you will have to know is LambdaMART, but to understand this model we have to briefly discuss its predecessors. 

The best resources I've found on this are a [Medium article by a Google EM](https://medium.com/@nikhilbd/intuitive-explanation-of-learning-to-rank-and-ranknet-lambdarank-and-lambdamart-fe1e17fac418) and a [blog post by Microsoft Bing Research](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/). There is also the original [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf) on LambdaMART and an overview [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) of the three models discussed below.

#### RankNET
RankNET is a neural network that tries to optimize a pairwise cross-entropy loss function. The idea is that the model should maximize the probability that an actual relevant document is ranked higher than an irrelevant document. It does this with a two-layer network that predicts the relevance score for each document. It then takes a pair of scores and appies the sigmoid function on the difference, and calculates the cross entropy with the true order, and backpropagates this loss. During prediction time, the documents are simply forward passed through the model and the output scores determine their ranking order. This [article](https://towardsdatascience.com/learning-to-rank-for-information-retrieval-a-deep-dive-into-ranknet-200e799b52f4) explains it pretty well.

#### LambdaRank
LambdaRank improves the train time of RankNET with a few mathematical tricks - basically, you can update the weights of the network by computing the gradients directly of the cost function to the scores and the scores to the weights instead of computing the cost function each time and backpropagating, which improves training time from quadratic run time to almost linear. The other major addition is weighting the loss function with the change in NDCG. This [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) goes into more detail.

#### LambdaMART
You can use the loss function from LambdaRank and apply gradient boosted trees (xgboost) to sequentially fit more trees on the gradients. This ends up yielding much faster run times, as tree models tend to forward pass much faster than neural nets. Additionally, it has the advantages of tree models of being able to handle mixed data types.
