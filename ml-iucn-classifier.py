import pandas as pd
import numpy as np

df = pd.read_csv("iucn-description-data.csv") #import data from the output of data mining code found at https://github.com/nikohartline/hartline-code-portfolio under the IUCN folder

import random
dftrim=df[["Conservation Actions","Population Trend"]]
dftrim=dftrim.dropna(how="any")
dftrim=dftrim.sample(frac=1)

text = dftrim['Conservation Actions']
target = dftrim['Population Trend']

# Perform feature extraction:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=False)
count_vect.fit(text)
counts = count_vect.transform(text)

# Train with this data with a Naive Bayes classifier:
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(counts, target)

#Try the classifier
keywords=['protect','reserve','ban','seed bank','agriculture','cropland','pastureland','hunting','urgent']
for i in keywords:
    print(nb.predict(count_vect.transform([i])))
