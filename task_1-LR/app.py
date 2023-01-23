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
import tqdm

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


def predict(input_file, model):
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    
    df = load_input(input_file)
    X = df['postText'] + df['targetParagraphs']
    X = X.apply(" ".join)
    X = X.to_frame(name="text")

    X["text"] = preprocessor.transform(X["text"])

    # Predict
    X["Y"] = pipeline.predict(X["text"])

    # Save the output
    for _, i in tqdm(X.iterrows()):
        yield {'uuid': i["uuid"], 'spoilerType': i["Y"]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.input, args.output)
