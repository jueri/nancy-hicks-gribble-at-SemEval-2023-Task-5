#!/usr/local/bin/python3
import argparse
import json
import pandas as pd
import numpy as np
from typing import List, Dict
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    with open(df, 'r') as inp:
         inp = [json.loads(i) for i in inp]
    return pd.DataFrame(inp)

    # if type(df) != pd.DataFrame:
    #     df = pd.read_json(df, lines=True)
    # return df


class ClassificationModel:
    def __init__(self):
        self.models = {
            "passage": AutoModelForSequenceClassification.from_pretrained("/models/roberta-news-full-3_passage-2022-12-22-T10-50-31", num_labels=2,local_files_only=True),
            "phrase": AutoModelForSequenceClassification.from_pretrained("/models/roberta-news-full-1_phrase-2022-12-21-T20-36-22", num_labels=2,local_files_only=True),
            "multi": AutoModelForSequenceClassification.from_pretrained("/models/roberta-news-full-3_multi-2022-12-22-T11-54-08", num_labels=2,local_files_only=True)
        }
        self.tokenizer = AutoTokenizer.from_pretrained("/models/roberta-news-full-3_passage-2022-12-22-T10-50-31",local_files_only=True)
        self.fields = {"passage": ['postText', 'targetTitle', 'targetParagraphs'], "phrase": ['postText'], "multi": ['postText', 'targetTitle', 'targetParagraphs']}

    def get_text(self, row, tag):
        text = ""
        for field in self.fields[tag]:
            if isinstance(row[field], list):
                text += ' '.join(row[field])
            elif isinstance(field, str):
                text += row[field]
            else:
                raise NotImplemented
        return text

    def predict_probability(self, text: str, model):
        tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**tokenized).logits
        return logits.argmax()

    def predict_one(self, row: str):
        probabilities = {}
        for tag_name, model in self.models.items():
            text = self.get_text(row, tag_name)
            probability = self.predict_probability(text, model)
            probabilities[tag_name] = probability

        return max(probabilities, key=probabilities.get)


def predict(file_path):
    df = load_input(file_path)
    uuids = list(df['uuid'])

    classifyer = ClassificationModel()

    for idx, i in tqdm(df.iterrows()):
        spoiler_type = classifyer.predict_one(i)
        # predictions.append(spoiler_type)
        yield {'uuid': uuids[idx], 'spoilerType': spoiler_type}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    print(args.input)
    run_baseline(args.input, args.output)
    # run_baseline('Data/webis-clickbait-22/validation.jsonl', 'dev_output.jsonl')