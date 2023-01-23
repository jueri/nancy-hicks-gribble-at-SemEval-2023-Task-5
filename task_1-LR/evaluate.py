import json
import pickle

import pandas as pd
from sklearn.metrics import f1_score


def load_input(df):
    with open(df, 'r') as inp:
         inp = [json.loads(i) for i in inp]
    return pd.DataFrame(inp)


if __name__ == "__main__":
    evaluate = load_input("../Data/webis-clickbait-22/validation.jsonl")
    y_evaluate = evaluate['tags'].explode()

    X_evaluate = evaluate['postText'] + evaluate['targetParagraphs']
    X_evaluate = X_evaluate.apply(" ".join)
    X_evaluate = X_evaluate.to_frame(name="text")

    with open("../preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("../model.pkl", "rb") as f:
        pipeline = pickle.load(f)

    X_evaluate["text"] = preprocessor.transform(X_evaluate["text"])

    y_pred = pipeline.predict(X_evaluate)

    print(f1_score(y_evaluate, y_pred, average='micro'))