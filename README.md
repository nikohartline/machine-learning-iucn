# Machine learning for insights into the IUCN Red List data

The data were mined using another set of code from my github:
https://github.com/nikohartline/hartline-code-portfolio/tree/master/IUCN%20Data%20Mining%2C%20Analysis%2C%20and%20Visualization%20Scripts/Python%20scripts%20and%20outputs

The output of the data mining provided information on the population trend of a species (increasing, decreasing, stable, or unknown) for over 28,000 species. This project aims to identify the effectiveness of a machine learning algorithm to predict based on the wording of descriptions of conservation actions whether a species will be increasing or in decline.

The algorithm vectorizes each entry for conservation action using Count Vectorizer from the scikit package. These vectors are used to develop a model with 16,000 of the data points used as training data and the rest used for testing accuracy of the model. The model had about 60% accuracy as compared to a dummy model (most frequent population trend) with around 30% accuracy.
