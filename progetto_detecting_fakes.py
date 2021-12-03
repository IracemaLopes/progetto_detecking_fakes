import pandas as pd
import numpy as np
import transformers
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer # for more info visit https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3
from sklearn.linear_model import PassiveAggressiveClassifier #for more info visit https://www.geeksforgeeks.org/passive-aggressive-classifiers/
from sklearn.metrics import accuracy_score, confusion_matrix# for more info visit https://classeval.wordpress.com/introduction/basic-evaluation-measures/ and visit also https://machinelearningmastery.com/confusion-matrix-machine-learning/
from sklearn.model_selection import train_test_split #train_test_split is a function in Sklearn model selection for splitting
# data arrays into two subsets: for training data and for testing data.
# With this function, you don't need to divide the dataset manually.
# By default, Sklearn train_test_split will make random partitions for the two subsets.
# creating 4 portions of data which will be used for fitting & predicting values.
#further info visit https://realpython.com/train-test-split-python-data/

print('importing the dataset')
df=pd.read_csv("news.csv")

print('checking how many columns and rows there are')
print(df.shape)

print('printing the first few rows')
print(df.head())

print('Getting the labels')
labels=df.label
print(labels.head())


print('splitting the dataset into training and testing sets')
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7) #random_state simply sets a seed to the random generator,
# so that your train-test splits are always deterministic. If you don't set a seed, it is different each time. further info visit the link https://newbedev.com/what-is-random-state-in-sklearn-model-selection-train-test-split-example.
#test_size is the number that defines the size of the test set. It’s very similar to train_size. You should provide either train_size or test_size.
# If neither is given, then the default share of the dataset that will be used for testing is 0.25, or 25 percent.
print('printing train test split to see if it is working')
print(x_train)
print(x_test)
print(y_train)
print(y_test)


print('DataFlair - Initialize a TfidfVectorizer')
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7) #TfidfVectorizer uses an in-memory vocabulary (a python dict)
# to map the most frequent words to features indices and
# hence compute a word occurrence frequency (sparse) matrix
# please visit this link https://www.etutorialspoint.com/index.php/386-tf-idf-tfidfvectorizer-tutorial-with-examples
print(' printing TfidfVectorizer')
print(TfidfVectorizer)



print('DataFlair - Fit and transform train set, transform test set')
#tfid is useful in solving the major drawbacks of Bag of words by introducing an important concept called inverse document frequency.
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
#to know more about fit.transform() and tranform() visit this link https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
print('printing tfidf_vectorizer.fit_transform(x_train) to see if it is working')
print(tfidf_vectorizer.fit_transform(x_train))
print('printing tfidf_vectorizer.transform(x_test) to see if it is working')
print(tfidf_vectorizer.transform(x_test))




print(' printing DataFlair - Initialize a PassiveAggressiveClassifier')
pac=PassiveAggressiveClassifier(max_iter=50) #max_iter = 50 Set maximum total number of iterations
pac.fit(tfidf_train,y_train)
#iteration is repeating identical or similar tasks without making errors is something that computers do well and people do poorly.
# Repeated execution of a set of statements is called iteration.



print('DataFlair - Predict on the test set and calculate accuracy')
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
cf=confusion_matrix(y_test,y_pred)
print(cf)
TN=cf[0,0]
FP=cf[0,1]
FN=cf[1,0]
TP=cf[1,1]

print("True positive value:", TP)
print("False positive value:", FP)
print("True negative value:", TN)
print("False positive value:", FN)