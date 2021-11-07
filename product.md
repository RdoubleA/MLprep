# Machine Learning System Design

This section describes how to answer open-ended case study questions. Usually, the interviewer will give you a problem that they want to build a machine learning model for. For example "I want to predict when a customer will default on their loan" or "I want to build a tool that recommends LinkedIn users that are most likely to respond to recruiter InMails". You play the role of the consultant, and your task is to walk them through the best way to approach the problem.

The best resource on this topic I've found is [Educative's Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview) course. These are my notes from the course.

- [Set up the problem](#set-up-the-problem)
- [Understand latency and scale](#understand-latency-and-scale)
- [Defining metrics](#defining-metrics)
- [Gathering training data](#gathering-training-data)
- [Feature engineering](#feature-engineering)
- [Model selection](#model-selection)
- [Model training](#model-training)
- [Online evaluation](#online-evaluation-and-iterative-improvements)
- [Case studies](#case-studies)
- [More resources](#more-resources)


## Set up the problem
The interviewer will typically tell you to solve a vague problem. It's very important to first ask questions to be able to formulate it as a clear ML problem. For example, if you are tasked to build a search engine, ask what kinds of queries will it answer? What does the current search engine show? What improvements are desired to enhance user experience? This gives insight into what type of input you should expect, what task the ML model will solve, and what the output might be.

## Understand performance and capacity
To help with model selection, we need to understand any constraints we'll be working with. This will help define the requirements of your system. Two constraints you should ask about are latency and scale. For latency, ask questions such as "how fast do we want to return relevant posts/search results". Questions for scale help determine how much data you'll be working with, such as "how many websites will we sift through with our search engine?" or "how much user browsing history do we have access to to predict most relevant ads?"

## Defining metrics
Think about offline metrics and online metrics. Offline metrics are metrics used while developing/training your model. So, more ML specific metrics such as AUC/F1 score for classifiers, or whatever is used for the type of model for this task. Online metrics and metrics that are important for indicating performance during live deployment. Within online metrics, you should consider component metrics and end-to-end metrics. Component metrics are online metrics specific to your model and the task your model will be completing. If you model is ranking results for a search query, you can use NDCG (Normalized Discounted Cumulative Gain, or the sum of relevance scores for all results discounted by position and divided by the DCG of the ideal order). End-to-end metrics are more of topline metrics of the entire system your model is plugged into. So more global metrics such as user engagement, retention rate, etc.

## Gathering training data
Make sure you detail steps for getting quality training data, and a lot of it. Based on the task, you can get training data in three different ways:
1. Human labeled data. Probably the most expensive way to get data in terms of time and money. If you ask about what areas the current system fails, you could focus more on those samples.
2. Crowdsourcing. Only works if the data needs to be hand labelled and doesn't require specialized training.
3. Open-source datasets. Know some common open-source datasets for common ML tasks
4. Data collection from pre-existing system. There likely already is a current naive approach for the task you are trying to solve. You can utilize this and a user's interactions with it to gather more data. Most often you will use this.

### Dataset augmentation
This is commonly asked for image datasets. For images, you can always augment by translating, rotating, adding noise, changing brightness or contrast, etc. If there's a certain object you are trying to detect you can move that object, change its size, etc.

For non-image datasets, like for recommendation systems, you can ask the user more questions such as what their interests are when setting up a new account. Or observe user behavior such as what they name their spotify playlists and the associated songs. This would fall under number 4 for gathering training data.

A third expensive option is to use generative models such as GANs to create more samples of the type that you need, or apply style transfer for more uncommon sample types (such as more rainy day images).

## Feature engineering
From this dataset, what would be the important features you would use? Think about all the actors / data sources in your problem. For examples, for ad ranking you have the user, the user's browsing history, and the ads themselves. Interrogate each of these agents to see what features you can come up with. From the user you can get demographics such as location, age, race (which is why I keep getting ads from a local Indian aunty cooking delivery thing that I will never click), etc. From the browsing history you can get clicks on links related to the topic of the ad. From the ads you can features related to what the ad is about, their target audience, location of company, etc. Also think about how these agents interact as well. So features such as embedding similarity between user search queries and ad text, or embedding similarity from previous ad clicks and current ad.

## Model selection
Depending on the scale of the dataset and the latency constraints, you can choose a simpler model (regression, SVM) or something more complex (NNs). Most often you will be dealing with an enormous dataset, or a very constrained latency, in which case you must consider how the model will be deployed on a distributed system, and use the funnel approach.

Consider the complexity of the model you are thinking of. There's a cost to the model training and a cost to the evaluation by the model. Regression models are quick to train and evaluate but may not be suited for complex tasks. Neural networks are expensive to train and evaluate, but if constraints allow it, it may be the best suited option. Tree models are somewhere in the middle.

This is why it's important to clarify what is acceptable performance / latency. Maybe 99% of requests should have max 500 ms latency. Also clarify how much capacity the model should have, such as 1000 queries per second.

Once you clarify that, you might realize that even if a simple linear regression can evaluate in microseconds, if it needs to run for 100 million documents it would take 100s to successfully process a search query. Know that scaling ML involves deploying on distributed systems, where shards can process samples in parallel.

### The funnel approach
You can see that even with distributed systems, if a task requires the complexity of a deep learning model, you still won't be able to achieve sub-second runtime, especially since the number of individual machines you can run the model on may be limited as well.

The funnel approach starts by applying the simplest model to all the data to filter down the number of samples to an appropriate size for a more complex model to run. Imagine for our search engine example we need to process 100 billion documents. We can first use a rule-based system or a simple heuristic for document selection to reduce to 100k documents. Then we have two stages of ranking. Stage 1 might employ logistic regression to determine whether a document is relevant or not. This can help us choose the 500 most relevant documents. The second stage can be a slightly more complex model since we're working with only 500 documents, so a model such as lambdaMART can be used to score each document and sort by rank.

## Model training
Pretty straightforward, just remember the ML basics such as 80/20 training/validation split, distributions of both sets, variance/bias, k-fold cross validation, and all the metrics you previously came up with.

## Online evaluation and iterative improvements
Remember the online metrics - component and end-to-end - that you just mentioned. If the system performance is substantially improved, as in statistically significant in an A/B test, that you can deploy it. The interviewer may ask what you would do if the model's performance begins to drop, be ready to talk about investigating potential sources of error, i.e., which samples / cases are giving the model most trouble, are there other factors such as geographical location or some holiday or social media phenomenon that is impacting the model or maybe the model's training data is now too old and needs to be updated.

### Change in feature distribution
The data the model is now seeing is different / comes from a different distribution than what the model was trained on. For example, you trained on search queries in the summer, but now the queries during the winter holidays are vastly different. Or the logic that computes the features online might differ slightly than what the model was trained on offline on historical data. For example, maybe for ad engagement prediction you had data 30 days back on ad impressions, but online you only have 7 days. It's always worth observing your features, plotting their distributions, and observing any discrepancies.

### Overfitting/underfitting
If your model performs poorly online but does well on training/validation, then it is likely overfitting. Use a test set to anticipate this offline, or if you already are using a test set, maybe the distribution is different than what you're seeing online. Underfitting can usually be detected in the offline phase.

### Identifying where the model is failing
The best way to improve the model is naturally to look at the samples the model does poorly on. You could investigate user behavior related to these hard samples and discover that you need a new feature that your model didn't previously capture. You may also notice that the hard samples are all of a similar type, and you didn't have many samples of that type in your training set. If working with a large scale system/model with multiple layers, such as in the funnel approach, again take your hard samples where your model is failing overall and observe each layer to identify which part is failing.

## Case studies

### Search ranking

#### Problem statement

> Design a search relevance system for a search engine

First ask clarifying questions, focusing on scope, scale, user.
- Should this handle general search queires or domain specific search queries?
- How many websites should we expect to search?
- How many queries per second should we be expected to handle?
- Will this model be deployed on a distributed system? How many machines/shards will be able to run the model?
- Is the searcher a logged in user? (will we have access to historical user data)


#### Overall architecture

Assuming this is a large scale problem with millions to billions of documents, we would need a layered approach where we filter the dataset at every layer with increasingly complex models to cut down computational cost. We can break down the architecture into the following stages:
- Document Selection
- Stage 1 Ranking
- Stage 2 Ranking

If it's not a large scale problem we might be able to shave off the top simpler layers and keep the complex models. This of course also depends on how much data you will have.

#### Dataset

When thinking about the training data, always start with identifying the actors in this scenario. For search, you have the searcher, the query, the documents, and context (browsing history, time of year, popular searches). Interrogate each to come up with features.
- Searcher: demographic information, interests
- Query: intent (usually calculated by embedding/deep learning model), historical use (if people have searched it before and what they ended up clicking)
- Document: location, page rank
- Context: time of the year, ongoing events

Now interrogate the interactions between these actors to come up with more features.
- Searcher + document: if searcher has engaged with similar documents before, distance between locations
- Query + document: similarity between embeddings/intent, TFIDF score, simple text match, clickthrough rate of other users using same query

The last thing to do is to come up with labels. Positive, negative, or relevance score. Binary labels would follow the pointwise approach, and scores would follow the pairwise approach (since common ranking models use a pairwise loss). For binary, you can label a document as a positive sample if the user clicked on it and spent x amount of time on the website. A negative sample would be a link that had an impression (was shown to user) but was not clicked. For scores, you can either use human raters (very expensive) or use historical user sessions to generate data. So, maybe the site with the longest time spent would have the highest relevance score, one with a short amount of time would have a smaller score, and documents that were not clicked have the lowest. With this you can easily amass a large dataset if the engine is used heavily.

#### Model selection
For document selection, we will use a simply heuristic to filter for the most relevant documents. For example, we can use a linear combination of text match/TFIDF, document popularity, intent match, personal data match, etc. and select the top 100k. Then we can start using our engineered features.

For stage 1 ranking, we want to focus on _recall_, or selecting as many of the relevant documents from all relevant documents as possible. For recall, simple linear models such as logistic regression is sufficient, especially when we want runtime to be very fast. Here we can use our binary labels from successful sessions. The loss function will be binary cross entropy, and the best way to evaluate it with an offline metric would be AUC. If the interviewer asks if dataset is imbalanced, you can counter with using F1 score instead, or subsampling the majority class. Also since we want to focus on recall that would also be a good metric to watch. We send maybe the top 500-1000 documents with highest probability to next stage.

For stage 2 ranking, since we have much less data to process, we can use a more complex model, such as LambdaMART (tree model) or even LambdaRank (neural network) depending on the answers to capacity and scalability. Since we're using ranking models, we can use our relevance scores we gathered before. Here we want to focus on _precision_ because we want to make sure the documents we rank highly are actually highly relevant. The offline metric we can use to evaluate this model is NDCG.

If the interviewer asks about filtering results, or maybe you can preemptively bring it up, just briefly discuss potentially using another simple ML model to detect profanity, offensive content, etc, you can even use the same training data, but with different labels provided by human raters or user reports.

#### Deployment
Now we deploy the model. Maybe talk through what you expect the run time to be based on how many machines in a distributed system you have access to. Online metrics you can use to evaluate performance can be topline metrics like clickthrough rate (clicks per impression), successful sessions (sessions with a click longer than certain amount of dwell time over total sessions), or low queries per session. This is where the interviewer might ask you how you can iterate on this and improve the model if it's failing. 


### Feed based system

### Recommendation system

Before we get into the methods, we should discuss how we measure a user's engagement with a movie. **Explicit feedback** is when a user provides a rating for something they've watched. This has some clear drawbacks: you will have missing not at random (MNAR) data because a user might only provide ratings for movies they liked and not movies they didn't like, or vice versa. Also they won't consistently provide ratings for every movie they watch. In general, it's

### Self-driving car and image segmentation

### Entity linking system

### Ad prediction

### Other examples
Here are some example case study questions that I've come across:

I want to create a tool for recruiters that suggests LinkedIn users that are most likely to be receptive to their recruitment efforts. How can I approach this?

As a gaming company, we want to continue to give players a fresh experience. We want to develop a system that recommends a new character to a player based on their playing style and the characters they've used before (Super Smash Bros, for example). What type of model could we use?

Tell me how you can develop a model to predict when a customer will default on their loan.

A university wants to detect fraud in their college applications, such as forgery, made up accomplishments, made up grades, etc. How can I build a fraud detection model to achieve this? How can you convince the director of admissions to use this model?

A cell phone company finds that it is much cheaper to pay for promotions / deals/ marketing efforts to retain customers that are close to churning (switching providers) than to lose the customer entirely. Build a model to predict whether a customer will churn or not so that the company knows when to begin redirecting promotions to retain that customer.

Build a model for a smart thermostat that can determine when to turn on or off a heater in a room.

## More resources
This is a great article for further reading: https://medium.com/@ushnish.de/the-machine-learning-product-interview-question-ac244c642ff0

It is very difficult to prepare for these questions other than doing practice. One way to practice is to read actual case studies by other companies. Here is a good resource for that: https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/case-studies.md

Here is a great video walking through an example case study: https://www.youtube.com/watch?v=kE_t3Mm8Z50