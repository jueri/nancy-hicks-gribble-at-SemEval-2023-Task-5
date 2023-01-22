#!/usr/local/bin/python3
import argparse
import json
import pandas as pd
import numpy as np
import nltk as nltk
import spacy
import regex
import re
import sklearn
from nltk import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 using a linear model that creates a spoiler type for each clickbait post.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def get_phrase(row, model_phrase):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    return [model_phrase.predict(question, context)['answer']]

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]

def preprocess(df):
    df = df[['postText','targetParagraphs']]

    # convert all columns into strings
    df[['postText', 'targetParagraphs']] = df[['postText', 'targetParagraphs']].astype(str)
    #tokenize the relevant columns (not actually used for the Bag of Word approach)
    tokenizer = RegexpTokenizer(r"\w+")
    df["postText_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["postText"]), axis = 1)
    df["paragraph_tokens"] = df.apply(lambda row: tokenizer.tokenize(row["targetParagraphs"]), axis = 1)

    #removing stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    df["postText_tokens"] = df.apply(lambda row: [element for element in row["postText_tokens"] if element not in stopwords], axis = 1)
    df["paragraph_tokens"] = df.apply(lambda row: [element for element in row["paragraph_tokens"] if element not in stopwords], axis = 1)
    
    #lowercasing 
    df['postText_tokens'] = df['postText_tokens'].map(lambda row: list(map(str.lower, row)))
    df['paragraph_tokens'] = df['paragraph_tokens'].map(lambda row: list(map(str.lower, row)))
    
    # multiple space to single space
    df[['postText_tokens', 'paragraph_tokens']] = df[['postText_tokens', 'paragraph_tokens']].replace(r'\s+', ' ', regex=True)
    #special characters
    df[['postText_tokens', 'paragraph_tokens']] = df[['postText_tokens', 'paragraph_tokens']].replace(r'\W', ' ', regex = True)

    #lemmatize tokens
    df['postText_tokens'] = df['postText_tokens'].apply(lemmatize_text)
    df['paragraph_tokens'] = df['paragraph_tokens'].apply(lemmatize_text)

    #count column lengths
    df['postText_length'] = ""
    df['paragraph_length'] = ""
    for i in range(len(df)):
        df['postText_length'][i] = len(df['postText_tokens'][i])
        df['paragraph_length'][i] = len(df['paragraph_tokens'][i])
    
    for i in range(0, len(df)):
        questionmark = "?"
        df['has_questionmark'] = 'posthasquestionmark'
        if questionmark in df['postText'][i]:
            df['has_questionmark'] = 'posthasnoquestionmark'
            
            
    mean_postText_length = df['postText_length'].mean()
    mean_paragraph_length = df['paragraph_length'].mean()

    df['postText_length'] = df['postText_length'].apply(lambda x: 'overavg_post_length' if x > mean_postText_length else 'underavg_post_length')
    df['paragraph_length'] = df['paragraph_length'].apply(lambda x: 'overavg_paragraph_length' if x > mean_paragraph_length else 'underavg_paragraph_length')
    
    for i in range(len(df)):
        df['has_numeric'] = any(str.isdigit(c) for c in df['targetParagraphs'][i])
    df['has_numeric'] = np.where(df['has_numeric'], 'hasnumeric', 'nonumeric')
   
    nlp = spacy.load('en_core_web_lg')
    df['Entities'] = df['postText'].apply(lambda sent: [(ent.text, ent.label_) for ent in nlp(sent).ents])  
    df['Entities'][0]

    for i in range(len(df)):
        tostring = str(df['Entities'][i])
        tostring = ' '.join(str(item) for tup in df['Entities'][i] for item in tup)
        df['Entities'][i] = tostring
  
    df['multi_signs'] = ""
    multi_signs = ['1.', '2.', '3.', '4.', '5.', '6.','7.', '8,', '9.', '10', 'first', 'second', 'third', 'list']
    df['multi_signs'] = df['targetParagraphs'].apply(lambda x: any([k in x for k in multi_signs]))

    df['combined_texts'] = ""
    df['combined_texts'] = df['postText'] + " " + df['targetParagraphs'] + " " + df['postText_length'] + " " + df['paragraph_length'] + " " + df['has_questionmark'] + " " + df['has_numeric'] + df['Entities']

    return df
    

def predict(inputs, model):
    for row in tqdm(inputs):
        try:
            
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
    model = PKl_Filename = 'Pickle_LR_Model.pkl'
    with open(PKl_Filename, 'rb') as file:  
        Pickled_LR_Model = pickle.load(file)
    
    args = parse_args()
    run_baseline(args.input, args.output, model = Pickled_LR_Model)
    #run_baseline(input_file="/Users/nicolasrehbach/Documents/GitHub/ANLP2223/Data/webis-clickbait-22/train.jsonl", output_file="test.jsonl", model_phrase = model, model_passage = model, model_multi = model_multi)