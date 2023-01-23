import numpy as np
import spacy
from sklearn.base import BaseEstimator


class Length(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for x in X.values:
            result.append([len(x[0])])
        return np.array(result)

# above mean
class AboveMean(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        length = X.apply(lambda x: len(x))
        self.mean = length.mean()
        return self

    def is_above(self, x):
        if len(x) > self.mean:
            return 1
        else:
            return 0

    def transform(self, X):
        result = []
        for x in X.values:
            result.append([self.is_above(x[0])])
        return np.array(result)

# Question
class Question(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        return self

    def has_question(self, x):
        if '?' in x:
            return 1
        else:
            return 0

    def transform(self, X):
        result = []
        for x in X.values:
            result.append([self.has_question(x[0])])
        return np.array(result)

# Has num
class HasNum(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for x in X.values:
            if any([i.isdigit() for i in x[0]]):
                result.append([1])
            else:
                result.append([0])
        return np.array(result)


# num entities
class NumEntities(BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en_core_web_lg")


    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        return self
    
    def count_entities(self, x):
        doc = self.nlp(x)
        return len(doc.ents)

    def transform(self, X):
        result = []
        for x in X.values:
            result.append([self.count_entities(x[0])])
        return np.array(result)


# Itterration
class Itterration(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y=None):
        return self
    
    def count_itterration(self, x):
        multi_signs = ['1.', '2.', '3.', '4.', '5.', '6.','7.', '8,', '9.', '10', 'first', 'second', 'third', 'list']
        return sum([1 for i in multi_signs if i in x])

    def transform(self, X):
        result = []
        for x in X.values:
            result.append([self.count_itterration(x[0])])
        return np.array(result)
