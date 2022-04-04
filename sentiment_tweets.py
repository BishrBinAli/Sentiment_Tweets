import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split


# File path containing the data
file_path = "data/text_data/Corona_NLP_train.csv"

# File containing stopwords
stopwords_file = "data/english-stop-words-large.txt"

# Directory to save plot
plt_dir = 'outputs/'

# Reading data from csv file
data = pd.read_csv(file_path, encoding='latin-1')

# Finding possible sentiments of a tweet
possible_sentiments = data['Sentiment'].unique()
print("\nPossible values of sentiment are: ", possible_sentiments)

# Finding second most popular sentiment in the tweets
sentiment_counts = data['Sentiment'].value_counts(ascending=False)
print("\nSecond most popular sentiment: {} with {} counts".format(sentiment_counts.index[1], sentiment_counts[1]))

# The date with te greatest number of extremely positive tweets
date_sentiments = data.groupby('Sentiment')['TweetAt'].value_counts()
print("\nThe date with the greatest number of extremely positive tweets: {} with {} counts".format(
      date_sentiments['Extremely Positive'].index[0], date_sentiments['Extremely Positive'][0]))


tweets = data['OriginalTweet']

# Converting the tweets to lowercase
tweets = tweets.str.lower()

# Replacing all non alphabetical characters with whitespaces
tweets = tweets.str.replace('[^A-Za-z]', ' ')

# Making all whitespaces into single whitespace
tweets = tweets.str.strip()
tweets = tweets.str.replace(' +', ' ')


# Converting tweets into list of words
words_tweet = tweets.str.split()

# Total number of all words in the corpus
words = words_tweet.explode()
print("\nTotal number of all words: ", len(words))

# Number of distinct words in the corpus
words_count = words.value_counts()
print("\nTotal number of distinct words: ", len(words_count))

# 10 most frequent words in the corpus along with their counts
print("\nThe 10 most frequent words in the corpus are:")
print(words_count[:10])

# Getting the stop words
with open(stopwords_file) as f:
    raw_verse = f.read()
f.close()

stopwords = raw_verse.split('\n')

# Removing the stopwords
words_stopwords_removed = words_count.drop(labels=stopwords, errors='ignore')

# Words with <= 2 characters
words_2char = words_stopwords_removed.index[words_stopwords_removed.index.str.match(r'\b\w{1,2}\b') == True]

# Removing the words with <= 2 characters
words_cleaned = words_stopwords_removed.drop(labels=words_2char, errors='ignore')

# Number of all words in the modified corpus
print("\nTotal number of all words after cleaning: ", sum(words_cleaned))

# 10 most frequent words in the modified corpus along with their counts
print("\nThe 10 most frequent words in the corpus after cleaning are:")
print(words_cleaned[:10])


# Computing fraction of documents in which each word occurs
vectorizer = CountVectorizer(vocabulary=words_cleaned.index)
X = vectorizer.fit_transform(np.array(tweets))
frac_words = X.getnnz(axis=0)/len(tweets)
frac_words = np.sort(frac_words)

# Plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# Line plot of word frequencies with normal scale
ax1.plot(frac_words)
ax1.set_xlabel("Words")
ax1.set_ylabel("Fraction of documents")
# Line plot of word frequencies with log scale in y-axis
ax2.plot(frac_words)
ax2.set_xlabel("Words")
ax2.set_ylabel("Fraction of documents")
ax2.set_title("Y-axis : log scale")
ax2.set_yscale('log')
# Saving plot
plt.savefig(plt_dir + 'word_fraction.jpg')
plt.show()


# Storing the tweets as numpy array
tweets = np.array(tweets)

# Producing sparse representation of term document matrix
# Using the cleaned words after removing stop words and words with less than 2 characters
vectorizer = CountVectorizer(vocabulary=words_cleaned.index)
td = vectorizer.fit_transform(tweets)

# Initialising Multinomial Naive Bayes classifier
clf = naive_bayes.MultinomialNB(alpha=0.5)

# Splitting to training and test set
X_train, X_test, y_train, y_test = train_test_split(td, data['Sentiment'], test_size=0.1)

# Train the classifier with the term document matrix and sentiment values
clf.fit(X_train, y_train)

# Predict on the test term document matrix with the classifier
predicted = clf.predict(X_test)

# Accuracy
acc = sum(y_test == predicted)/len(predicted)
print("\nAccuracy on the test set = ", acc)




