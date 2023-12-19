# Bayesian-Network-for-Tweet-Sentiment-Analysis

# Twitter Sentiment Analysis
NLP Sentiment Analysis on tweets using a Bayesian Network implemented from scratch without any library.

## Introduction
### Naive Bayes assumption of Independence
Naive Bayes emerges as an alternative to Joint Distribution, it is based on Bayes' assumption that all attributes are conditionally independent. This assumption, more formally called the Bayes Theorem, may seem like a crazy assumption, but it has become very powerful over time because it allows us to use computational power. Over the years it has become a standalone as a robust classifier that with little training and large databases can give many good results.

### Objective
For the project we are given a dataset which is more than 1M tweets.
These tweets have already been processed with a "Lancaster" stemming algorithm that extracts the lexeme from the words to reduce the size of the dataset and concentrate the entire workload on the morphemes.
For example, if we had the following words `<programmer, program, programmatically, programmable>` the Lancaster algorithm would reduce all words to one: `<program>`.
Our algorithm must classify tweets according to the sentimentLabel attribute, that is, it must predict whether a tweet is positive (sentimentLabel = 1) or negative (sentimentLabel = 0).
We need to look at how they affect the size of the training dictionary and the train-test split.
Finally, we will also need to examine and implement some smoothing method to avoid the probabilities at 0.

## Proposal
The proposal is incremental, meaning that we will first start with a basic bayesian network and then we will be adding more features and capabilities to our model.

### Basic Bayesian Network
To implement a first Bayesian network, I followed the steps below:
-  **Feature engineering**: In this step we will read the data, do a quick analysis to detect anomalies and divide it into train and test sets.
-  **Train the model**: In this section we will generate a dictionary with the words that we find in the training set. We will get rid of meaningless words with the help of stopwords.
We will also calculate the probabilities of each word.
-  **Predictions**: We will test our training with the new data set, test. This is where we need to apply smoothing if we don't want to ignore words that aren't part of the dictionary.
-  **Metrics**: Once we have done the classification we will be able to extract different metrics to determine the performance of our model.

#### Feature Engineering: Train-test split
##### Read dataset
In this step we read the dataset and get rid of the columns <tweetId, tweetDate> since we will not use them for anything.
We will also eliminate the rows where we find some NaN to avoid possible a posteriori complications, 22 rows in particular are insignificant considering that we have more than 1M of "memories".
##### Train-split test
We will specify the percentage of the training set.
To do this in a balanced way, we will first separate the dataset according to the target sentimentLabel variable, thus obtaining two subsets, the positive and the negative.
#### Training: Dictionaries generation and "stopwords"
##### Dictionaries generation
Model training will be done by calculating the probabilities of each word.
To do this we must first calculate the frequency of each word according to its target attribute. We will save it in a dictionary called dict_words.
We will use this dictionary of counts to create the dictionary where we will calculate the probabilities of each word in the dictionary. We will save it in a dictionary called prob_cond.

##### Stopwords
The NLTK library's stopwords feature has been used to detect words that have no meaning such as pronouns or prepositions.
Stopwords is available in many languages, which is why an auxiliary function has been made to analyze the language of tweets (detect_lang).

#### Prediction
With dictionaries we can start making predictions.
For each tweet, we look up the words in our dictionary of probabilities and multiply them.
With the help of Bayes's assumption, we'll calculate the largest conditional probability accumulation for each word in the tweet so we can predict which outcome it belongs to, whether the tweet is positive or negative.
In this section, we will ignore words that do not appear in the training dictionary. We ignore them to avoid generating probabilities of 0.

#### Metrics
Given  `True Positive (TP)`, `True Negative (TN)`, `False Positive (FP)` and `False Negative (FN)` we can define some metrics as so:
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1Score = 2 * (Recall * Precision) / (Recall + Precision)
```

Accuracy tells us the success rate of our model.
In binary classifiers, as is the case, the accuracy metric can be misleading when we have an unbalanced dataset in which there are more positive than negative tweets or vice versa.
But in this case, if we calculate the percentage of each outcome in the database we see that the dataset is balanced so the accuracy will be a good metric.

### Crossvalidation
Model validation is a widely used technique for quickly detecting overfitting, when the model is too close to training data or underfitting, when the model fails to achieve the minimum and necessary learning in training.

It also serves to validate that the results are consistent and reliable for different train-test splits of the data.

(*) Keep in mind that cross-validation is done using only the entire training set, doing so with the testing set would be a mistake.

#### K-Fold technique
A good technique that allows us to do cross-validation without a large load of resources is K-fold.

This technique divides the training dataset into k parts called folds. Of these k folds, 1 will be test and the remaining k-1 will be training. K experiments will be done so each fold will be tested once and finally we will get the crossvalidation score as the average of the accuracy obtained

To implement this first a specific function has been done in which the train dataset is divided into K partitions and with a loop we will make sure that each partition is used as a test once.

### Playing with dictionary sizes
To reduce the size of the dictionary, a function is created in which the word counter dictionary is sorted from most to least frequent and then with a given percentage as a parameter we split, leaving the percentage of words. more frequent.

In this case mida_dict will be the value as a percentage of the most frequent words we want to keep.

With sorted we do the descending order of the doubles of appearance, both positive and negative, and with the help of itertools we can get rid of a part of the dictionary.

#### Accuracy and runtime
As you can see, we have a balanced set, so Accuracy is a good estimator of the model's performance.

To carry out the study of this section, a getMetricsDictTrain function has been carried out which, with a list of different dictionary sizes and training set, will perform tests with the combination of all sizes and we will save accuracy and execution time. of each possible combination of values.
### Smoothing Techniques
So far we are ignoring words that do not appear in our dictionary. But there must be some method to quantify the rest of the words as well.
So what do we do with words that are not in our dictionary?
In the naive bayes function there is a "smoothing" parameter that depending on the value we give it will apply (or not) any of the following settings:
- `smoothing = “None”`
  
  Unknown words will be ignored to avoid generating probabilities of value 0.

- `smoothing = “laplace_add1”`

  Apply a simple "laplace add 1" smoothing technique that assumes all
words have been viewed with standard frequency.
  In our case we have seen it 2 times, 1 for each possible outcome of the target sentimentLabel variable.
  In our case the value of the unseen words in the generated dictionary will be:
  ```python
  dict_words = { "<word>": [1, 1] }
  ```

- `smoothing=“laplace_smoothing”`
  
  Apply the Laplace Smoothing technique with the alpha parameter, as explained in
class.
  
