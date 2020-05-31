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

Central limit theorem states that if you take an adequate number of samples from a population, regardless of the actual probability distribution of the populations, your sample means will approach a normal distribution. This means with repeated sampling from a population where the distribution is unknown, which is most real world distributions, we can calculate confidence intervals, conduct hypothesis testing, on the sample mean since it is normally distributed and make inferences about the population. This allows us to conduct t-tests with sample means to see if the population mean is nonzero, for example. At least 30 samples is considered an adequate number of samples for the means to approach normal distribution.

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

### Activation Functions

Activation functions are necessary to introduce nonlinearities to neural networks to learn more complex functions and patterns. Otherwise, neural networks become a fancy linear regression.

Sigmoid – constrained output, fully differentiable, but tails can lead to vanishing gradients. Primarily used at final layer for binary classification

Tanh – constrained output, larger range for large gradients compared to sigmoid, but also has vanishing gradients

ReLU – computationally efficient, no vanishing gradient in linear range, may enforce a bit of sparsity. BUT, dying neurons when they enter negative regime. Most common activation function, this is a strong default option to use

Leaky ReLU – ReLU, except negative regime replaced by a slight slope, typically 0.01x. Fixes dying ReLU problem. Especially used in GANs where gradient flow is very important.

Softmax – sigmoid, generalized to multiple classes. Usually for numerical stability, we subtract the max of x from the exponent, which ensure you have large negative exponenets instead of large positive exponenets, which prevents overflow


### Bias vs Variance Tradeoff

Model error due to bias is a result of underfitting, or that the model is too simple and makes too many prior assumptions about the data to accurately fit it. Examples of high bias models are linear regression, since it assumes the data is linear, for example. If we increase model complexity we can find more complex patterns, but that leans towards overfitting the data and causes the model to generalize poorly. This is error due to variance, meaning the model is too complex and is fitting noise in the data instead of the general pattern. You can think of it as the model varies highly when changes datasets and generalizes poorly.

Thus, whether to increase model complexity or decrease model complexity is a tradeoff between bias and variance. High bias means better generalization but poorer fit, but high variance means excellent fit but poor generalization. The ideal model is somewhere in the middle.

You can determine whether a model has high bias or variance by examining the training and validation/test errors. A larger validation error than the training error indicates the model generalizes to unseen data poorly and is overfitting the training data, therefore it has high variance. If both training and validation error are high, then the model is underfitting the data and has high bias. There are many methods to address these issues:

Underfitting – expand the model capacity (add more trees, hidden layers, dimensionality, etc), add more input features, remove regularization

Overfitting – use regularization, decrease model capacity, use dropout (in neural networks), get more training data (helps model distinguish signal from noise and generalize to new data better), use ensemble methods, remove input features, early stopping
