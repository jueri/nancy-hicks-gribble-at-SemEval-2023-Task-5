import nltk
import numpy as np
import spacy
from nltk import WordPunctTokenizer
from sklearn.base import BaseEstimator


class RemoveStopwords(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        nltk.download("stopwords")
        self.tokenizer = WordPunctTokenizer()
        self.stop_words = nltk.corpus.stopwords.words("english")
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            tokenized_text = [
                token
                for token in self.tokenizer.tokenize(sentence)  # if len(token) > 2
            ]
            text = " ".join(
                [token for token in tokenized_text if token not in self.stop_words]
            )
            results.append(text)
        return np.array(results)


class Lemmatize(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        # python -m spacy download de_core_web_sm
        self.nlp = spacy.load(
            "en_core_web_lg"
        )  # prepare language model if needed only
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            text = " ".join([token.lemma_ for token in self.nlp(str(sentence))])
            results.append(text)
        return np.array(results)


class Lowercase(BaseEstimator):
    def get_feature_names(self):
        return [self.__class__.__name__]

    def fit(self, X, y):
        return self

    def transform(self, X):
        results = []
        for sentence in X:
            text = sentence.lower()
            results.append(text)
        return np.array(results)