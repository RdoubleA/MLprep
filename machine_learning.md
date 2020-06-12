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

The most important hyperparameters for random forests are the number of trees and the number of features to use for each tree. Increasing the former usually improves performance upto a certain point, but increases computation time. This also decreases variance. Increasing the latter increases variance and reduces bias, decreasing it will reduce variance and increase bias.

Increasing depth of each tree increases variance.

Increasing the minimum number of samples at each leaf node forces a regularization effect because it prevents the tree from becoming too complex. It increases bias.

https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

### Boosting

While bagging randomly samples a dataset and trains many learners in parallel, hoping to reduce variance of each learner by combining the votes of many uncorrelated learners, boosting reduces bias by sequentially building new learners on the mistakes of past learners. Thus, boosted trees are more prone to overfitting than random forests and are also more difficult to tune, but generally see better performance. Additionally, random forests have slower run time due to the high number of trees. Random forests are generally more effective for datasets with a lot of noise. Boosted tree ensemble methods usually perform better on datasets with less noise and handle imbalanced datasets well. 

#### AdaBoost

Adaboost sequentially builds on weak learners by changing the weights of samples - more weight for misclassified and less weight for correctly classified - and resampling according to those weights to fit new learners. Thus, new learners build on the mistakes of the previous. Trees contribution to final decision are weighted by how well they performed.

#### Gradient Boost

Gradient Boost does the same except by fitting trees to the residuals of the last tree, and using a learning rate to weigh every tree's contribution to the final decision.

#### XGBoost

Extreme gradient boosting is a specific implementation of gradient boost that adds a few key features that make it significantly more powerful than vanilla gradient boost. These are regularization, parallelization, and the second derivative of the loss function. XGBoost uses a second-order Taylor series expansion on any loss function (MSE, BCE) as the objective function. Each new split or tree is evaluated in terms of the gain, or how it minimizes this objective function, and the regularization, which is a measure of how complex the tree is. Tree complexity is calculated through the values on the leaves and the number of leaves. The math behind XGBoost is complicated, I point you towards the official documentation for XGBoost for more info: https://xgboost.readthedocs.io/en/latest/tutorials/model.html

Essentially, XGBoost improves on traditional graidne tboost by making it faster through parallelizing, addressing overfitting through controlling each additional tree complexity, and redefining objective function with addition of second order derivatives. XGBoost generally performs the best out of all the tree-based methods, and is widely used in Kaggle.


### Accuracy Metrics

Accuracy is the total correct divided by all total classifications. Or, (TP + TN) / (TP + FP + FN + TN). Useful only when classes are balanced. If imbalanced, accuracy is not meaningful, i.e., detecting cancer that happens at a 1% rate in a population. You can just say no all the time and get a 99% accuracy.

Precision measures the proportion of predicted positives that were truly positive. That is, TP / (TP + FP). In the cancer example, we never predicted a true positive, thus precision is 0. Precision is useful when we want to be very sure of our prediction. For example, we don’t want to falsely diagnose someone with cancer, so we optimize for precision. Increasing precision decreases our chances for Type I error, or when we wrongly predict positive for an actual negative result. However, since we’re taking extra care to be precise, we may let a lot of cases that actually have cancer slip by undetected, which could also be problematic. In other words, precision will maximize true positives, minimize false positives, but may not minimize false negatives. Chance of Type I error decreases but chance of Type II error will increase. Depending on your problem, this may be ok.

Recall measure how many actual positives were correctly classified. It is useful when we want to make sure to capture all the positives. For example, if we are trying to predict terror threats, we want to capture all potential threats even if it may be a false alarm. Maximizing recall minimizes chance for Type II error, or missing a threat that was actually there. However, if you only maximize recall, you can get a recall of 1 if you predict 1 for every example. Thus, recall will minimize false negatives and maximize true positives but may inflate false positives and decrease true negatives. Chance of Type II error will decrease, but chance of Type I error will increase. 

Precision and recall are two opposing forces. You want to be sure when you predict cases with cancer, but at the same time you want to be able to identify all cancer cases as much as possible, thus the tradeoff between precision and recall. Precision vs recall is the same tradeoff of Type I vs Type II error. F1 score measures how well the model does in all cases of this tradeoff, as it is a harmonic mean between the two. You can also use domain knowledge if you want to weigh finding all the positive cases more or being more confident in your positive classification more by the F1 beta score, which adds a factor to change the balance between the two.

https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226

Sensitivity is the true positive rate, or how many of the actual positives were correctly classified. It is equivalent to recall, statistical power, and the inverse of Type II error. Specificity is the true negative rate, or how many of the actual negatives were correctly classified.

Both of these come into play when using AUC ROC as a classification metric. When you vary the probability threshold used for classifying an example as positive, you change the model’s sensitivity and specificity. Increase the threshold (in the range of 0.5 - 1.0), and you get fewer positives, which will decrease sensitivity, FPR, recall and increase specificity, precision. Decrease the threshold and you will get more positives, which will increase sensitivity, FPR, recall and decrease specificity, precision. This is plotted by an ROC curve, which plots sensitivity against FPR, or 1 – sensitivity, or the fraction of negatives that are incorrectly classified, over various thresholds. The area under this curve measures a model’s ability to separate classes. An AUC close to 1 indicates that the model is able to perfectly distinguish classes. An AUC close to 0.5 means the model cannot distinguish classes at all. An AUC of 0 means the model is completely reversing the classes. In multi-class problems, you compute ROC curves for one class vs all for each class. AUC is a great metric for general performance of a model irrespective of the threshold used. However, when the costs for false negatives or false positives are imbalanced, or the classes are heavily imbalanced, then AUC is not that informative. F1 score is more appropriate for imbalanced cases, or area under precision and recall curve.

https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

### Class Imbalance

When classes are heavily skewed (90% yes, 10% no), the model can have trouble learning the minority class and predicting future minority class examples. To address this, you can:
 - Undersample the majority class and/or oversample the minority class (bootstrapping, repetition, etc)
 -  Weigh the minority class more in the cost function
 -  Use a more appropriate accuracy metric, such as AUC of precision-recall curve, F1 score

### Cross Validation

Cross validation is a validation method that partitions the dataset into folds, where all but one fold is used for training and the last fold is used for validation. This validation fold is rotated among all the folds and the model’s validation accuracy is averaged across all these experiments. Cross-validation is preferred over 

### Feature Importance

In linear models, you can use the coefficients to compute feature importance, or p-values from linear regression. In tree-based models, feature importance is calculated by Mean Decrease in Impurity, or sum of number of splits including that feature weighted by the number of samples it splits. One drawback of MDI is it assigns higher importance to high cardinality features, especially continuous variables, so other methods such as permutation importance are also used.

### Data imputation

When you have missing data, you can approach it the following ways:
- Replace all values with the mean/median value for that feature. However, this does not take into account the correlations between features and reduces the variance of this feature. It also does not work for categorical variables
- Replace values with the mode. This only works for categorical variables. Again, the issues with mean replacement also apply
- Remove samples with missing values. Only works if you have enough data.
- Train a model to predict the missing value. kNN is commonly used, to predict missing features based on the similarity of other features with other data points. Or a generative model.
- Sample from the rest of the dataset


### Hyperparameter search

Grid search - select some values for each hyperparameter and try every combination, like a grid

Random search - select a range of values for each hyperparameter and randomly choose values

Random allows you to try more different values in case one hyperparameter does not improve objective function at all - in which case grid search will be redundant for different values of the dud hyperparameter

Some parameters you should sample on a log scale instead of uniformly at random, especially learning rate which can vary on a log scale (0.0001 to 1). For example, sample the exponent uniformly at random and then use learning rate = 10 ^ that exponent. OR for values that range from 0.9 to 0.999, use 1 - 10 ^ that exponent

Bayesian optimization uses performance of previous searches to find the next best one, and tends to perform better than random, grid, or manual search. You can implement in Python using hyperopt.

Andrew Ng's lecture on this topic: https://www.youtube.com/watch?v=AXDByU3D1hA

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

An excellent reference: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiDy-TtqPjpAhXaJzQIHQ5KDxoQFjAYegQIDhAB&url=http%3A%2F%2Finfolab.stanford.edu%2F~ullman%2Fmmds%2Fch9.pdf&usg=AOvVaw1mHhOKehTffby-_BRMvvrY

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

### What if the distribution of test data is different from distribution of training data?

This will make models harder to perform well on the test data set. It will have high accuracy on the training set but not the test set, meaning it will overfit. You can mitigate this by mixing some samples from tthe test distribution with the training distribution and making sure the validation set matches the test set distribution as close as possible, otherwise it's a moving target for the model. Cross-validation with different splitting can help as well.

You might also consider using high bias models that can generalize better.

Differences in distribution may occur when training samples are obtained in a biased way, or maybe a different time period with different circumstances.

### How do I deal with outliers?

L1/lasso regularization can increase bias of the model and is also resistant to outliers. L1/MAE loss is more resistant to outliers than L2/MSE. Tree-based methods also tend to be more resistant than regression methods.

You can also transform the data (i.e., BoxCox transform) or remove outliers or Winsorize (clip) the outliers

### Feature engineering

Raw data is usually not in the proper format for a machine learning model to train effectively on it. Categorical variables need to be encoded for certain models. Some features have skewed distributions, and you might consider log transforming them so that the model can work with a wider range of values. There may be outliers that you may want to remove or cap / Winsorize. Missing data needs to be either removed or imputed with the mean value, or some other data imputation method. Anything that has to do with data cleaning is feature engineering.


https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

### Clustering

#### K-means clustering

Initialize a predetermined number of centers. Calculate cluster membership for each data point by computing distances to every center and picking the closest one. Recalculate new centers by finding the mean data points for each cluster. Repeat.

Advantages: Runs in linear time complexity, O(n)
Disadvantages: Need to determine number of cluster beforehand, different results every run due to random initialization

#### Mean-shift clustering

Set a number of circular sliding windows with radius r and randomly initialize their centers. Compute the mean of all the points within the sliding window and move the window to the new center point. Repeat for all sliding windows and continue until convergence. Remove overlapping windows at the end, and you have your clusters identified.

Advantages: No need to select number of clusters, automatically discovers this
Disadvantages: Need to select radius r, number of sliding windows

#### DBSCAN

Choose a random point, a neighborhood distance epsilon, and a minimum number of points. If there is at least the minimum number of points within the neighborhood, begin clustering. Mark this point as visited, and go to next unvisited point within this cluster. Continue until all points within this cluster is visited. Go to next unvisited point after this cluster is complete. If not enough points to start cluster, then label point as noise.

Advantages: works with arbitrary shapes of clusters, deals with outliers, do not need to predefine number of clusters
Disadvantages: does not work well when clusters have different densities

#### Gaussian Mixture model with EM

Choose a number of clusters and randomly initialize parameters for a normal distribution for each cluster (mean and std). Compute each point's probabilities for being in each cluster. Use these probabilities to update distribution parameters using EM. Continue until convergence.

Advantages: unlike K-means, can detect elliptical shaped clusters. supports mixed membership
Disadvantages: does not work as well with arbitrary shapes of clusters

#### Agglomerative hierarchical clustering

Initialize each data point as its own cluster. Compute average linkage between each pair of clusters, or average distance from points in one cluster to another. Combine two clusters with the smallest average linkage. Repeat until all data points in one cluster.

Advantages: detects hierarchical structure, allows freedom to choose number of clusters, works well with any distance metric
Disadvantages: O(n^3) time complexity

Great resource: https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68