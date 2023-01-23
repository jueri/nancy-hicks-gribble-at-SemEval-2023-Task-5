#!/usr/local/bin/python3
import argparse
import json
import pandas as pd
import re 
import numpy as np
import torch
import transformers 
from transformers import BertForQuestionAnswering, BertTokenizer, AutoModel, AutoTokenizer
from transformers import pipeline
from tqdm import tqdm

class Qa_model:
    def __init__(self, model_name, num_answers = 1):
        self.model_name = model_name
        self.num_answers = num_answers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_name, local_files_only=True)
        self.model = pipeline("question-answering", self.model_name, tokeniezer = self.tokenizer, max_length=500, truncation=True, return_overflowing_tokens=True, stride = 128, top_k= self.num_answers)
                     #pipeline("question-answering", "deepset/roberta-base-squad2", tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2"), max_length=500
                      #              , truncation=True, return_overflowing_tokens=True, stride=doc_stride, top_k=5)

    def predict(self, question, context):
        answer = self.model(question = question, context = context)
        return answer


def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def get_phrase(row, model_phrase):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    return [model_phrase.predict(question, context)['answer']]


def get_passage(row, model_passage):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    answer = model_passage.predict(question,context)['answer']

    candidates = []
    for sentence in context.split('.'):
        if answer in sentence:
            candidates.append(sentence.strip())
    
    if not candidates:
        print('No candidates found')
        return ['']
    elif len(candidates) == 1:
        return [candidates[0]]
    elif len(candidates) > 1:
        print('Multiple candidates found')
        return [candidates[0]]


def get_multi(row, model_multi):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    current_context = context
    results = []
    try:
        for _ in range(0,5):
            #current_context = current_context
            candidates = model_multi.predict(question, current_context)[0]
            current_result = candidates['answer']
            results.append(current_result)
            current_context = re.sub(current_result, '', current_context)
    except:
        print("Error generating multipart spoiler")
        results = "Error"
    return results
    

def predict(inputs, model_phrase, model_passage, model_multi):
    for row in tqdm(inputs):
        if row.get('tags') == ['phrase']:
            answer = get_phrase(row, model_phrase)

        elif row.get('tags') == ['passage']:
            answer = get_passage(row, model_passage)
        
        elif row.get('tags') == ['multi']:
            answer = get_multi(row, model_multi)
        else:
            print("Tag not found")
            raise NotImplemented

        yield {'uuid': row['uuid'], 'spoiler': answer}


def run_baseline(input_file, output_file, model_phrase, model_passage, model_multi):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
         inp = [json.loads(i) for i in inp]

         for output in predict(inp, model_phrase, model_passage, model_multi):
            out.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    model = Qa_model("/model")
    model_multi = Qa_model("/model", 5)
    args = parse_args()
    run_baseline(args.input, args.output, model_phrase = model, model_passage = model, model_multi = model_multi)
    #run_baseline(input_file="/Users/nicolasrehbach/Documents/GitHub/ANLP2223/Data/webis-clickbait-22/train.jsonl", output_file="test.jsonl", model_phrase = model, model_passage = model, model_multi = model_multi)