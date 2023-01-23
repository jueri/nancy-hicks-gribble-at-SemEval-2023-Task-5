from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from feature import Length, AboveMean, Question, HasNum, NumEntities
from preprocessing import RemoveStopwords, Lemmatize, Lowercase
import pandas as pd
import argparse
import json
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()

def load_input(df):
    with open(df, 'r') as inp:
         inp = [json.loads(i) for i in inp]
    return pd.DataFrame(inp)

def get_preprocessing_pipeline():
    pipeline = Pipeline([
        ("RemoveStopwords", RemoveStopwords()),
        ("Lemmatize", Lemmatize()),
        ("Lowercase", Lowercase())])
    return pipeline

def get_feature_pipeline():
    unigrams_feature = FeatureUnion(transformer_list=[("unigrams", CountVectorizer())])
    tfidf_feature = FeatureUnion(transformer_list=[("tf-idf", TfidfVectorizer(min_df=10, ngram_range=(1, 2)))])

    column_trans = ColumnTransformer(
        [
            ("unigrams", unigrams_feature, "text"),
            ("tfidf", tfidf_feature, "text"),

            ("Length", Length(), "text"),
            ("AboveMean", AboveMean(), "text"),
            ("Question", Question(), "text"),
            ("HasNum", HasNum(), "text"),
            ("NumEntities", NumEntities(), "text"),
        ],
        remainder="drop",
        verbose=True,
    )

    pipeline = Pipeline(
        [
            ("preprocessing", column_trans),
            ("classify", LogisticRegression(n_jobs=1, C=1e5)),
        ],
        verbose=True
    )
    return pipeline



def main(input, output):
    X = load_input(input)
    y = X['tags'].explode()
    X = X['postText'] + X['targetParagraphs']
    X = X.apply(" ".join)
    X = X.to_frame(name="text")
    
    print("start preprocessing")
    preprocessor = get_preprocessing_pipeline()
    X["text"] = preprocessor.fit_transform(X["text"])
    with open("preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)

    print("start training")
    pipeline = get_feature_pipeline()
    pipeline.fit(X, y)

    # save the model
    with open(output, 'wb') as f:
        pickle.dump(pipeline, f)





if __name__ == '__main__':
    args = parse_args()
    main(args)
    # main(input="Data/webis-clickbait-22/train.jsonl", output="./model.pkl")