# Machine Learning System Design

This section describes how to answer open-ended case study questions. Usually, the interviewer will give you a problem that they want to build a machine learning model for. For example "I want to predict when a customer will default on their loan" or "I want to build a tool that recommends LinkedIn users that are most likely to respond to recruiter InMails". You play the role of the consultant, and your task is to walk them through the best way to approach the problem.

The most important thing to practice is developing a framework for approaching the problem. Once you know an organized approach, you can tackle any question. Here, I go over the approach you want to take.

The best resource on this topic I've found is [Educative's Grokking the Machine Learning Interview](https://www.educative.io/courses/grokking-the-machine-learning-interview) course. There is also articles from an [ML engineer at Twitter](https://towardsdatascience.com/how-to-answer-any-machine-learning-system-design-interview-question-a98656bb7ff0), [ML engineer from Pinterest](http://patrickhalina.com/posts/ml-systems-design-interview-guide/), and [some guy on Leetcode](https://leetcode.com/discuss/interview-question/system-design/566057/Machine-Learning-System-Design-%3A-A-framework-for-the-interview-day).

- [Set up the problem](#set-up-the-problem)
- [Understand constraints](#understand-constraints)
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

## Understand constraints
To help with model selection, we need to understand any constraints we'll be working with. This will help define the requirements of your system. Two constraints you should ask about are latency and scale. For latency, ask questions such as "how fast do we want to return relevant posts/search results". Questions for scale help determine how much data you'll be working with, such as "how many websites will we sift through with our search engine?" or "how much user browsing history do we have access to to predict most relevant ads?". Also ask about hardware constraints to clarify how fast this model should be expected to run and how much storage it might need. 

## Defining metrics
Think about offline metrics and online metrics, you should be able to provide at least one of each. Offline metrics are metrics used while developing/training your model. So, more ML specific metrics such as AUC/F1 score for classifiers, or whatever is used for the type of model for this task. Online metrics and metrics that are important for indicating performance during live deployment. Within online metrics, you should consider component metrics and end-to-end metrics. Component metrics are online metrics specific to your model and the task your model will be completing. If you model is ranking results for a search query, you can use NDCG (Normalized Discounted Cumulative Gain, or the sum of relevance scores for all results discounted by position and divided by the DCG of the ideal order). End-to-end metrics are more of topline metrics of the entire system your model is plugged into. So more global metrics such as user engagement, retention rate, clickthrough rate, etc.

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

## Model architecture
Most problems involve a large dataset and you need to provide the most relevant items to a user. For these problems, use the **funnel approach**. The funnel approach involves two main stages: candidate selection and ranking. Candidate selection first filters the dataset for any item that is relevant. This stage will be focused on recall, meaning its goal is to get as many relevant items from the dataset that are actually relevant. This stage should use linear models or even simple heuristics, since they will be dealing with a large amount of data and they need to quickly pull out relevant items. For example, for a search query you can use logistic regression to first narrow down the entire corpus of data to a smaller amount that's relevant to the query, which will serve as the candidate items. The second stage, ranking, provides the top k most relevant candidates to the user. Since this is the final output to the user this model should be high precision, meaning out of the relevant positive examples, our model should output the actual most relevant items as close as possible. This might involve a heavier model like a tree ensemble or a neural network to provide scores to each item.

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

> Design a search relevance system for a search engine

#### Clarifying scale
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
> Design a system that will provide the most relevant tweets to a user

#### Clarifying scale and problem statement
- How many daily active users are there?
- How many tweets per day do we ingest?
- How fast should the feed be loaded? How many times per day does a user refresh the feed on average?
- Do we have access to user engagement history on the app? Their profile data?
- How does the feed system currently work?

The ML objective is this: build a model that takes in all tweets as an input and predict probability of engagement for each tweet, and send top k to a user in order of relevance.

#### System architecture
Since we have potentially millions of tweets per hour, we will do a staged approach. The feed ranking will be performed in three stages. 
1) The first stage will filter for tweets that are connected to the user's social graph since those have higher likelihood of engagement, and we will achieve this with a simple heuristic.
2) The second stage will involve a linear model to retrieve the relevant tweets out of the social graph. 
3) The third stage will use a tree-based model or neural network to rank the relevant tweets in order of relevance, and this will be the final output to the user

#### Metrics
We should think of two metrics: at least one for offline evaluation and at least one for online evaluation. Bonus points for thinking of _counter-metrics_, or negative engagement
- Online: clickthrough rate (when user clicks on thread), conversions (user follows new profile), time spent on thread, likes/comments/retweets, reports/hiding tweets (negative)
	+ You can assert that this could be dependent on business objectives. Maybe Twitter wants higher overall engagement, or higher time spent on app
	+ Alternatively, use a linear combination of likes/comments/retweets/reports
- Offline: stage 2 - F1 score since dataset unbalanced (many impressions, not many clicks), or if you downsampled negative samples you can use AUC but keep in mind you may have to recalibrate model's probabilities. stage 3 - similar

#### Feature engineering
Start with all the actors in this scenario and their interactions:
1) User
2) Tweet
3) Author
4) Context

- User (not much help here)
	+ Demographics, profile info like age, region
- Author
	+ Page rank for how influential they are in social graph, number of followers
	+ recent engagement rate (interactions divided by views) over past three months
	+ topic engagement rate (perhaps as a histogram vector). deduce topic from tweet content using NLP models, or hashtags
- User-Author
	+ tweet similarity - create tf-idf vectors of bag of word embeddings for all tweets they posted or engaged with, aggregate via sum or average, and calculate similarity between their embeddings
	+ topic similarity - make a histogram vector of topics they've engaged with, calculate similarity
	+ historical engagement - how many replies/likes/retweets in past three months between them
	+ social graph - common followees
- User-Tweet
	+ tweet similarity - calculate similarity between user vector and tweet embedding vector
	+ social graph - users followees have all engaged with this tweet
- Tweet
	+ length, contains video, time posted
	+ historical engagement - interactions in past hour, day, etc
- Context
	+ day, time, any holidays, season, any current events

#### Training data
Use the existing system and online user engagement to gather training data. When a tweet is shown to the user but no like/comment/retweet, then that's a negative sample. Otherwise, positive sample. Since user engages with very low percentage of tweets, we will need to downsample the negative samples to have an even split of positive and negative. If we still have enough data after that, then AUC for our classifiers should be sufficient, we just need to calibrate the probabilities afterwards. On the other hand, if data is not sufficient, we can keep all negative samples but use F1 score instead to address class imbalance. Assuming we have plenty of data, we can go with even split.

For train/test split, it would be good to capture user's behavior across the entire week. So two weeks of training data should be enough, and since we want to predict future engagement, it would be ideal to split train/test on time dimension. So, first two weeks for training and use third week for test for a 66/33 split.

#### Model training
1) No model for first stage, just take recent tweets (at max maybe 2 days ago) from user's followees and tweets they've engaged with
2) Can train several different models. A first pass could be logistic regression with L1 regularization and a tree model such as random forest or gradient boosted tree. It would be good for feature selection early on with L1 regularization and the feature importance from the tree models (num of splits with the feature weighted by number of samples it splits), in case we need to remove features due to overfitting or to try to improve performance. We should experiment with hyperparameters such as regularization type, regularization weight, number of trees, min samples per leaf, etc based on performance on training and validation set. Then we select model with the best AUC score.
3) The same applies here, except you might be trying tree models and neural networks. Should experiment with hyperparameters such as number of layers, size of layers, learning rate, etc.

#### Model evaluation
If there was a prior system, you can A/B test with the new model vs the old system. You can perform statistical tests on the user engagement rate, or the online metrics mentioned (clickthrough, time spent on tweets, likes/comments/retweets). If its significant and the effect size is reasonable, you can deploy it, BUT also consider if the complexity, latency, memory constraints have significantly changed and you can decide based on that tradeoff.

### Recommendation system
> Build a model that displays recommended movies to a user

#### Clarification
- How many users/movies do we have?
- How are users accessing the movies? (is it phone app, browser, tv app, etc)
- How fast does it need to display recommendations?
- What is the business objective?
- Is there a current system for recommending movies?

The ML objective is to output a list of movies that has the highest probability of user engagement.

#### Model architecture
1) Candidate generation. Out of entire corpus of movies, use collaborative filtering, content-based filtering, and embedding similarity to generate user-movie scores. Take the top K movies. Recall that collaborative suffers from cold start (hard to calculate scores for new users), but does not require domain knowledge, embeddings also suffer from cold start for both users and movies AND requires domain knowledge, but can generate more robust embeddings, and content does not have cold start (if you onboard) but requires domain knowledge. Can just use all three.
2) Ranking. Use logistic regression, trees, or neural network to predict probability of user engagement and rank by highest probability
3) Re-ranking. Not model based, but modifying the recommendations to bring more diversity, introduce past watched or remove past watched.

#### Metrics
- Offline
	+ Mean average precision, mean average recall, F1 score for second stage
	+ for first stage you could use something like recall to validate your collaborative/content/embedding methods
- Online
	+ session watch time (time spent per session)

During training, we will selection model that performs best on F1 score. Then, we will test whether to deploy model using A/B testing on session watch time

#### Data
The actors are user, movie, context
- User
	+ demographics (age, location, language)
	+ genre engagement histogram, actor engagement histogram
	+ average session time
- Movie
	+ actors, genre, tf-idf vector of basic details like director, producer, etc
	+ length, language, country
	+ recent engagement in past few hours, days
	+ date posted, rating
- Context
	+ holiday, season, time of day, day of week
- User-movie
	+ embedding similarity
	+ collaborative score, content-based score
	+ similarity between past watched movie embeddings and current movie
	
Positive label is a watch, negative is an impression but no watch. Use negative downsampling due to class imbalance.
	
#### Model training
Use logistic regression to predict probability and track F1 score. Train on four weeks of data to capture weekly patterns - first two weeks is training, third week is validation, fourth is test. Logistic regression is great if we want to prioritize speed and space or have limited data, but if it suffers from high bias we can use random forest/gradient boosted trees because that can automatically learn feature interactions and nonlinear relationships. Based on tradeoff between run time, storage, variance, bias, we can upgrade further to a two-layer neural network. LR and trees have high explainability if that is valued, but neural networks do not. NN is great if there is a lot of data and less limitations on speed.

#### Model evaluation
A/B testing on session watch time.

### Self-driving car and image segmentation

### Entity linking system

### Ad prediction
> Build a system that shows ads that is most likely to be engaged with by the user

#### Clarifications
- How many ads, advertising companies?
- How large is the user base?
- Does this need to be real time, i.e., ranked when the site is loaded, query is entered? Or can we rank offline? How fast does it need to be?
- Do we have data on user history?
- What is the current system of showing ads?
- What type of content is in the ad? Image, video, text?

The ML objective is this: given a set of ads, train a model that ranks them in order of highest likelihood of user engagement, with these latency/capacity/scale constraints/requirements.

#### System architecture
We can employ a two-staged system:
1) Ad selection. This simply takes a subset of the full library of ads based on a simple heuristic, such as similar location, same topic as user's query, or maybe advertisers want to target a specific demographic or interest. We've have a SQL database indexed by this, and entries point to ad storage location on blob storage.
2) Ad prediction. Once we have our initial subset, we can train a simple model to predict probability of engagement. A model such as logistic regression, or random forest/GBT should suffice. The output of this model will determine the order which we can present ads

#### Metrics
To determine success for our system, we need offline metrics to tune its performance and online metrics to determine if we want to deploy it. 
- Offline
	+ Mainly for stage 2. Typically for logistic regression we would use AUC or F1, but both of these don't penalize how far off the prediction might be from a positive label, and we might need that granularity for ranking ads since we also have to incorporate the bid amount. Instead, you can use the raw cross-entropy loss.
- Online
	+ Clickthrough rate, downstream action rate, overall revenue
- Counter
	+ ad reports, ad hides
	
#### Data
The actors in our problem are the user, the ad, the advertiser, and context
- User
	+ demographics, interests
- Ad
	+ target demographics, target interests
	+ word embeddings (bag of words, tf idf, word2vec)
	+ image embeddings (VGGnet)
	+ Past engagement over 3 months, impression count so far
	+ bid
- User-Ad
	+ Average of prior engaged ad embeddings (word, image) and its similarity with this ad
	+ Topic engagement histogram (3% of clicks were sports, 5% was tech, etc)
	+ Similarity of query with ad text
- Advertiser
	+ industry, location
	+ region wise engagement histogram
- User-Advertiser
	+ prior engagement (bought from this place before, recently visited site)
- Context
	+ season, time of day, day of week
	
We can gather training data from user engagement in the existing ad system. A click will be a positive example and an impression with no click can be negative. We can downsample the number of negative samples to balance the dataset, just remember to recalibrate the model's predicted probabilities.

#### Model training
1) This is just a simple retrieval of ads from db via a topic or interest
2) Train a logistic regression model using log loss as the metric to determine the optimal model. 

#### Model evaluation
Use A/B testing on clickthrough rate to determine whether our model is worth deploying.

For online learning, you can engineer the system to continuously collect data in real time and then retrain and redeploy the model at every fixed time interval.

### Other examples
Here are some example case study questions:
- Create a tool for recruiters that suggests LinkedIn users that are most likely to be receptive to their recruitment efforts
- Develop a system that recommends a new character to a player (Super Smash Bros, for example)
- Predict when a customer will default on their loan
- Develop a model to detect fraud in college applications. How can you convince dean of admissions to use this? How can you explain it to them?
- Build a model to predict whether a customer will churn (stop using the product or subscription. Think anything like Curology, online courses with subscription, Netflix, Spotify, etc)
- Build a model for a smart thermostat that can determine when to turn on or off a heater in a room
- Recommend artists/albums to users on Spotify
- Recommend restaurants to users in Google Maps
- Recommend add-on items in a shopping cart
- Predict which app a user will click on next on a smartphone
- Create a system for word prediction on a smartphone keyboard/search engine/word document as the user is typing
- Predict which comment a user is most likely to enage with on a group post
- Predict the best time to send a push notification/email to a user
- Build a system to predict the value of homes on Airbnb

You can also peek this [github page](https://github.com/chiphuyen/machine-learning-systems-design/blob/master/content/case-studies.md) for more case studies.
