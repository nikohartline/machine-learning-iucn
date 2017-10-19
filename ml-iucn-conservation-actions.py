#Python code for exploring machine learning methods to better understand the relationship between the wording of species-specific conservation actions in the IUCN red list of endangered species on the population trend of the species listed. Trends are: Increasing, Decreasing, Stable, and Unknown

import pandas as pd
import numpy as np

df = pd.read_csv("iucn-description-data.csv") #import data from the output of data mining code found at https://github.com/nikohartline/hartline-code-portfolio under the IUCN folder

import random
dftrim=df[["Conservation Actions","Population Trend"]]
dftrim=dftrim.dropna(how="any")
dftrim=dftrim.sample(frac=1)

text = dftrim['Conservation Actions']
target = dftrim['Population Trend']
print(len(target))
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=False)
count_vect.fit(text)
counts = count_vect.transform(text)

#Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# A random set of 16,000 species out of the ~28,000 available as training data
nb.fit(counts[0:16000], target[0:16000])

predictions = nb.predict(counts[16000:])
correct_predictions = sum(predictions == target[16000:])
incorrect_predictions = (len(target) - 16000) - correct_predictions
print('# of correct predictions: ' + str(correct_predictions))
print('# of incorrect predictions: ' + str(incorrect_predictions))
print('Percent correct: ' + str(100.0 * correct_predictions / (correct_predictions + incorrect_predictions)))

from sklearn.dummy import DummyClassifier
db = DummyClassifier(strategy='most_frequent')

db.fit(counts[0:16000], target[0:16000])

# Dummy classifier to show the relative benefit of ML methods for population trend prediction based on the IUCN's
dummy_predictions = db.predict(counts[16000:])
correct_dummy_predictions = sum(dummy_predictions == target[16000:])
print('Percent correct: ', 100.0 * correct_dummy_predictions / (len(target)-16000))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(nb, counts, target, cv=20)
print(scores)
print(scores.mean())
