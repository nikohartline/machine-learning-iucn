#Since the idea of certain words being more important may influence the classifier's accuracy, we add in a term frequency-inverse document frequency weighting scheme and test the validation accuracy herein.
import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/iucn-description-data.csv") #import data from the output of data mining code found at https://github.com/nikohartline/hartline-code-portfolio under the IUCN folder

import random
dftrim=df[["Conservation Actions","Population Trend"]]
dftrim=dftrim.dropna(how="any")
dftrim=dftrim.sample(frac=1)

text = dftrim['Conservation Actions']
target = dftrim['Population Trend']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

SGDmodel = SGDClassifier()

from sklearn.model_selection import cross_val_score, cross_val_predict

text_model = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
                      ('tfidf', TfidfTransformer()),
                      ('SGDmodel', SGDClassifier()),
 ])

scores = cross_val_score(text_model, text, target, cv = 10)
print(scores)
print("the average cross validation accuracy of the tf-idf classifier was: %.1f%%" % int(100*scores.mean()))
