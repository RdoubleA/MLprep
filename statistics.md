# Probability and Statistics

- [Long-tailed Distribution](#long-tailed-distribution)
- [Confidence Interval](#confidence-interval)
- [Central Limit Theorem](#central-limit-theorem)
- [Measures of Central Tendency](#measures-of-central-tendency)
- [Standard deviation vs standard error](#standard-deviation-vs-standard-error)
- [Correlation vs Covariance](#correlation-vs-covariance)
- [p-values](#p-values)
- [Type I vs Type II error](#type-i-vs-type-ii-error)
- [Statistical power](#statistical-power)
- [Probability vs Likelihood](#probability-vs-likelihood)
- [MLE vs MAP](#mle-vs-map)
- [MLE for linear/logistic regression](#mle-for-linear-logistic-regression)
- [Combinations and Permutations](#combinations-and-permutations)
- [Bayes Theorem](#bayes-theorem)
- [Regression to the mean](#regression-to-the-mean)
- [Probability Distributions](#probability-distributions)
- [A/B and MAB testing](#a-b-and-mab-testing)
- [Coefficient of determination](#coefficient-of-determination)
- [Uncorrelated but not independent random variables](#uncorrelated-but-not-independent-random-variables)
- [Common probability problems](#common-probability-problems)
- [Further reading](#further-reading)

## Long-tailed Distribution

Long tailed distributions are probability distributions where a small number of values have very high frequency / probability and a large number of values have low frequency / probability, resulting in a long tail in the graph of the distribution that slowly tapers off.

Real world examples of this are social networks, where a majority of people have a relatively small number of a friends, and fewer and fewer people have a very large number of friends. Another examples is sales of different products, some products may be the most popular and drive a majority of the sales, whereas most of the products may have much fewer sales.

## Confidence Interval

Confidence intervals are a range of numbers that likely contains the population mean we are looking for. A 95% confidence interval means there is a 95% chance the interval contains the population mean (specifically, it means that the procedure used to create the interval, i.e., sampling and the math for making the interval, can create an interval with a 95% chance of containing the parameter)

Calculating the confidence interval involves the Central Limit Theorem. Because of CLT, our sample mean becomes a random variable that is normally distributed. Thus, we can calculate the range of values that will fall within 95% of the sampling distribution for our 95% confidence interval. This interval may contain the actual population mean with 95% confidence, since 95% of intervals we construct with this method will actually contain the population mean.

Z is the critical value of a standard normal distribution that lies at 95% of the area under the distribution, which is (-1.96, 1.96). If the population variance is known you can use that, where sigma is the STD of the population. If not, as in most cases, you will use the t-distribution and the sample standard error.

Some common mistakes:

95% confidence interval contains 95% of the population values – FALSE, 95% of the population values could be estimated from the sample. The 95% means that 95% of the intervals we make with this random sampling process will contain the population mean

95% confidence interval has a 95% chance of containing the population mean – kinda true, the fine print is that the procedure creates intervals that contain the population mean 95% of the time. An interval has the mean or it doesn’t

95% confidence interval of a larger sample size is more likely to contain the population mean than a smaller sample size – Nope, both intervals have an equivalent chance of containing the population mean since calculating confidence interval with standard error takes sample size into account. The main advantage of a larger sample size is narrowing that range.

This is a great resource for testing your understanding: http://www2.stat.duke.edu/~jerry/sta101/confidenceintervalsans.html

## Central Limit Theorem

Central limit theorem states that if you take an adequate number of samples from a population, regardless of the actual probability distribution of the populations, your sample means will approach a normal distribution, with a mean equivalent to the population mean and standard deviation according to standard deviation of population divided by square root of sample size. This means with repeated sampling from a population where the distribution is unknown, which is most real world distributions, we can calculate confidence intervals, conduct hypothesis testing, on the sample mean since it is normally distributed and make inferences about the population. This allows us to conduct t-tests with sample means to see if the population mean is nonzero, for example. At least 30 samples is considered an adequate number of samples for the means to approach normal distribution.

## Evaluating a linear regression model

When you fit a linear regression model, you are trying to estimate the coefficients that related X to Y plus some error term. This error term is a random variable, and linear regression assumes that this is normally distributed with the same variance for all samples. Thus, your coefficients are also random variables with some unknown mean (the true coefficient) and the variacne from the error term. When you fit the model, you are sampling the coefficient from its distribution centered around the true coefficient. Then, you can determine statistical confidence in this predicted value by estimating the variance (some calculation from the sum of squares of residuals) then computing a t-test where the null hypothesis is that &beta; = 0. The p-value will then tell you how likely the predicted &beta; was if the null hypothesis is true and there was not relationship between X and Y.

Other metrics you can use are the coefficient of determination (R squared) which assesses how much variance is captured by the model, or the residuals themselves. If you plot the residuals, they should appear random and not be skewed one way, otherwise there is some relationship in the data your model is not capturing.

This StackExchange [link](https://stats.stackexchange.com/questions/352383/how-to-calculate-p-value-for-multivariate-linear-regression) has a helpful explanation.

## Measures of Central Tendency

Mean – average of all examples
Advantages: all data points contribute to calculation, thus it is more sensitive when data points change or new points are added. Best used when distribution is data is near symmetric
Disadvantages: very sensitive to outliers, not ideal for long-tailed distributions for described the typical/middle value since it will be skewed towards outliers

Median – middle example when samples are arranged in order
Advantages: less sensitive to outliers, good for non-symmetric distributions (ex: income, house prices, where disparity is high)
Disadvantages: agnostic of distribution of data

Mode – most common element
Advantages: best used for categorical data where mean and median cannot be calculated
Disadvantages: not informative for continuous data


## Standard deviation vs standard error

Standard deviation describes the dispersion of the samples, and is agnostic of the distribution of samples or the population mean. Standard error measures how far the sample mean is from estimating the population mean. It is smaller than standard deviation, since it decreases as we increase number of samples (s / sqrt(n)). Standard error is used to construct confidence intervals, standard deviation is part of the calculation but it is not a metric itself in describing how accurate our estimates of the population mean are.

## Correlation vs Covariance

Both measure how a change in one variable affects the other, but correlation is normalized by the random variables standard deviation to range -1 to 1, while covariance is unbounded

## p-values

A p-value is the probability of observing the statistic as extreme as it is given the null hypothesis is true.

## Permutation tests

Randomize the group labels. Calculate the difference between means. Repeat this a large number of times (10k, 100k). Plot the distribution of differences. Where does the original difference fall? Reject or accept null hypothesis based on the probability of original difference.

## Type I vs Type II error

Type I error is what occurs when you reject the null hypothesis even though it is true in actuality. If we consider rejecting the null hypothesis as positive, then this would be a false positive. The probability of a Type I error is given by the p-value. For example, if you have a p-value of 0.19, that means the statistic you observed is 19% likely to be observed if the null hypothesis is true. If we reject the null hypothesis in this case, we have a 19% chance of being wrong. Precision of a classifier measures how well a model minimizes Type I error.

Type II error is when you fail to reject the null hypothesis when the null hypothesis is false. If rejecting the null hypothesis is a positive example, then this is a false negative. It is related to the power and sensitivity /recall of a test / classifier, as sensitive classifiers / tests with high statistical power minimize false negatives.

## Statistical power

Power is the probability of not making a Type II error. It is the probability we can accurately detect the signal and accurately reject the null hypothesis. It is similar to sensitivity.

When significance level is increased, you loosed the threshold for rejecting a null hypothesis. Therefore, you are more likely to accurately reject a null hypothesis, but more likely to wrongly reject a true null hypothesis. Thus, increasing the significance level increases your power, decreases your chance for Type II error, but increases chance for Type I error. If you decrease signifiance level, you are making it more strict to reject a null hypothesis. Thus, your power decreases, your Type II error increases, and your Type I error decreases.

Other ways to increase your statistical power are to increase sample size (generally a good thing), have less variance in your data (not under your control), or have a true population parameter further than what the null hypothesis states (i.e., the signal you are trying to detect is very strong, also not under your control).

## MLE vs MAP, posterior vs likelihood

Let's say we are trying to find the parameters &theta; for a model that will predict Y from data X. If you've done a lot of machine learning, you've commonly seen **maximum likelihood estimation** (MLE), where you solve for &theta; that maximizes the output labels:

![](https://latex.codecogs.com/gif.latex?L(\theta)=\prod_i^N&space;p(y|x;\theta))

This probability is the **likelihood** of &theta;. Notice that theta is _after_ the pipe symbol, basically saying that how likely are these parameters when we observe our data. This same probability is the posterior of Y because it is the probability of Y given some evidence. This works when &theta; is fixed and is not a random variable (which is why it comes after the semicolon and x is after the pipe, because x is a random variable but theta is not). 

If we want to model it as a random variable, and add some constraints as a prior, then we use **maximum a posterior estimation**. 

![](https://latex.codecogs.com/gif.latex?argmax_{\theta}P(\theta|X)=argmax_{\theta}\frac{P(X|\theta)P(\theta)}{P(X)})


Now we are interested in the **posterior** of &theta;, and theta is _before_ the pipe symbol. This is also different from MLE because now we have a prior P(&theta;), and we need to find the _distribution_ of &theta;. You don't see this often for estimating &theta;, because most often we are not modeling &theta; as a random variable. If we did have information on what sort of distribution &theta; takes, then MAP is our tool to use. MAP is typically used in Bayesian frameworks, such as Naive Bayes. But actually in Naive Bayes we are performing MAP on Y, not theta.

Actually, when you maximize the above equation and assume the probability of theta is uniform (meaning we don't know what values our parameters will take), you reach the same equation as MLE. Thus, MLE is a type of MAP assuming uniform distribution of the random variable in question.

A great [source](https://agustinus.kristia.de/techblog/2017/01/01/mle-vs-map/) on this goes into more detail. 


## MLE for linear/logistic regression



Know that minimizing the cost function for linear regression assumes the following:
1. Observations of X in the training set are independently, identically distributed
2. Residuals (errors from predicted to actual) are normally distributed with zero mean and the same variance for all examples


## Combinations and Permutations

Groups with the same elements but different orderings are **multiple different permutations** but are all the **same, one combination**

$$C=\frac{n!}{k!(n-k)!}$$

$$P=\frac{n!}{(n-k)!}$$

where $n$ is the number of objects you are choosing from, and you choose $k$ objects

Further reading: https://medium.com/i-math/combinations-permutations-fa7ac680f0ac

## Bayes Theorem

$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$

This is applied whenever we are given probabilities for certain conditions that are related. This is also the basis for the Naive Bayes classifier. This should be **memorized**

## Regression to the mean

Further observations following extreme values are closer to moderate values, i.e., if two parents are taller than average than their next child will likely be shorter than them, or closer than average. Of course, this assumes that the predominant factor is chance / luck.

## Probability Distributions

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

Used when you want to find out the probability of a certain number of successes, n, after a given number of events, x. Thus, we first must find the number of combinations with the desired number of successes and multiply by the probability of any one of those combination from happening. Thus:

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

## A/B, MAB, and backtesting

A/B testing is the same as a simple experiment where you manipulate one variable and keep all others constant. It is used when you want to see whether one change impacts a metric that you care about. It requires you to randomize you A and B populations such that there are no covariates and both populations are equally distributed as possible.

A/B testing is used to evaluate the statistical significance of a change on a metrics(s). It is useful when you want to make a business decision with confidence. Sometimes you need to do a power analysis to determine the minimum number of samples you need to make a decision.

Advantages: 
- evaluates statistical significance
- can use multiple metrics
- want to evaluate learnings from experiment about both A and B cases for future business decisions

Disadvantages:
- loss of traffic / certain metrics in the worse variation (A or B), opportunity cost of running the experiment
- need a long period of time to reach statistical significance, not good for time-sensitive decisions
- not effective when there are more than one alternative option, because then you need to run for every pair
- not effective when there is low traffic

Backtesting is when you flip the control and the variant after the initial test. This is to ensure the effect observed from the original test was due solely to the change and nothing else. If the backtest reduces the effect by an equal amount, then the change worked.

Multi-arm bandit testing is a form of dynamic optimization - as the test is run, MAB reallocates traffic to the best option at every time step instead of keeping the population sizes fixed during the entire experiment. As a consequence, you cannot evalute statistical significance, but you can optimize on a certain metric, so it's good for quick evaluations of a change or multiple changes without gathering the statistical support.

Advantages:
- Optimizes an important metric (e.g., conversions) instead of statistical learnings
- good for time-sensitive decisions, low traffic
- good for multiple variations

Disadvantages:
- no statistics to support a superior variation with confidence
- cannot evaluate more than one metric

## Coefficient of determination (R squared)

R^2 is a value that measures how much of the variance in the data is captured by the model, and it is generally used for regression models. It is calculated by 

$$R^2=1-\frac{SSR}{SST}$$

where SSR is the sum square of the residuals and SST is the sum square of the differences of every point from the actual mean.

A high R^2 is not always good. It could indicate overfitting. Some models where the residuals are systematically over or under the regression line and are not randomly distributed around it could have a high R^2.

A low R^2 is not always bad. You can still have statistically significant variables with a low R^2. Your data could be inherently noisy. Or maybe you don't need to predict values with precision, in which case a low R^2 isn't too bad.


## Uncorrelated but not independent random variables

Remember that correlation only captures linear relationships between variables. So if one variable is the square of another, they are not correlated but they are certainly dependent. Independent means there is no relationship between the variables whatsoever, linear or nonlinear.

## Sampling

## Common probability problems

#### St. Petersburg paradox

> A coin is flipped until heads appears for the first time, at which point the player is rewarded $2n where n is the number of flips to achieve the first heads. How much would you bet on this game?

Use the geometric distribution, which is used to model probability until first success, P(X=x) = (1 - p)^(x - 1)p. The expected value of this random variable is 1/p, so we expected heads to appear first after two flips and our reward should be $4. Thus, you would bet less than $4.

#### Die to generate 1-7

#### Coupon collector problem

#### Overlapping intervals


#### Airplane probability problem

> 100 passengers board a plane with 100 assigned seats. Everyone has a ticket with an assigned seat number. One person has lost their ticket and decides to sit in a random seat. Every other passengers sits in their assigned seat unless it is taken, in which case they sit in a random seat. What is the probability that a passenger will sit in their assigned seat?

This link has a great explanation: https://medium.com/i-math/solving-an-advanced-probability-problem-with-virtually-no-math-5750707885f1

#### Penney's game

This game usually involves three coin flips, but is commonly asked with two.

> Players A and B play a coin flipping game where if one of them gets their chosen combination that player wins and the game ends. Two coins are flipped until either player A or B win. A wins if HH is flipped. B wins if TH is flipped. Who would you bet on and why?

There are four possible combinations: HH, HT, TH, TT. Two of the combinations will end the game because a player wins. Now think through the possibilities if the other two, HT and TT were flipped. In both of these scenarios, either a head is flipped next, giving the game to player B, or a tail is flipped, in which case player B will win no matter what because a head will eventually be flipped after a tails. Thus, A only wins if HH is first flipped, a 1/4 probability, but B wins in every other scenario, a 3/4 probability.

If B chose HT as their combination, then both are equally likely. The scenario TH puts both players halfway to victory, and the next flip is 50% chance for either of them, and TT puts neither clsoer to victory.

This resource is a great explanation: http://www.math.unl.edu/~sdunbar1/ProbabilityTheory/BackgroundPapers/Penney%20ante/PenneyAnte_CounterintuitiveProbabilities.pdf

## Further reading

Probability cheatsheet: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiZuaWj1OvpAhVWs54KHQJkCwsQFjACegQIBBAB&url=https%3A%2F%2Fstatic1.squarespace.com%2Fstatic%2F54bf3241e4b0f0d81bf7ff36%2Ft%2F55e9494fe4b011aed10e48e5%2F1441352015658%2Fprobability_cheatsheet.pdf&usg=AOvVaw0ZN07X1cQMxVdnDVPPOzz-

Use this! https://github.com/JifuZhao/120-DS-Interview-Questions/blob/master/probability.md. Also this guide is really great in general.

