import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from datasets import Dataset
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorWithPadding
import wandb
from datetime import datetime
import os

def load_dataset(
    fields: List[str], 
    files: Dict[str, str]) -> pd.DataFrame:
    
    mapping: Dict[str, int]={'passage': 0, 'phrase':1, 'multi':2}

    dataset = {}
    
    def encode_label(label: str):
        return mapping[label]

    def load_data(file: str):
        df = pd.read_json(file, lines=True)

        data = []
        for _, i in df.iterrows():
            text = ""
            for field in fields:
                if isinstance(i[field], list):
                    text += ' '.join(i[field])
                elif isinstance(field, str):
                    text += i[field]
                else:
                    raise NotImplemented

            data.append({
                "text": text,
                "label": encode_label(i["tags"][0])})
        return data


    for split in list(files.keys()):
        dataset[split] = load_data(files[split])

    return dataset


def preprocess_dataset(dataset: List[Dict[str, List[str]]], model_base: str, model_name: str):
    if model_base.startswith("roberta"):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

    elif model_base.startswith("bert"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)     

    else:
        raise NotImplementedError    

    # create tokanizer function with correct tokenizer
    def preprocess_function(sample):
        return tokenizer(sample["text"], truncation=True) 

    # process slices
    dataset_tokenized = {}
    for slice in list(dataset.keys()):
        slice_dataset = Dataset.from_list(dataset[slice])
        slice_tokenized = slice_dataset.map(preprocess_function, batched=True)
        dataset_tokenized[slice] = slice_tokenized

    return dataset_tokenized


def create_model(model_base: str, model_name: str, dataset):
    # Get num labels
    num_label = len(dataset[list(dataset.keys())[0]].unique("label"))

    if model_base.startswith("roberta"):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)

    elif model_base.startswith("bert"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)     

    else:
        raise NotImplementedError   

    # Metrics
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Model
    training_args = TrainingArguments(
        output_dir="./Checkpoints/"+MODEL_NAME+"-"+now,
        logging_dir='./LMlogs', 
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        report_to = "wandb",
        logging_steps = 10,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        evaluation_strategy='epoch',
        save_strategy='epoch',)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer


def config_name(model, fields, mapping):
    now = datetime.now().strftime("%Y-%m-%d-T%H-%M-%S")
    if sum(mapping.values())> 1:
        type_ = "multiclass"
    else:
        type_ =  [k for k, v in mapping.items() if v == 1][0]
    return model+"_"+str(len(fields))+"_"+type_+"_"+now


if __name__ == "__main__":
    LOGGING = True
    MODEL_BASE = "roberta"
    PROJECT_NAME = "PAN-classification"

    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    data_paths = {
        "train": "Data/webis-clickbait-22/train.jsonl", 
        "validation":"Data/webis-clickbait-22/validation.jsonl"}

    # Grid search
    models = ["roberta-base", "roberta-news-full"]
    field_config = [["postText"], ["postText", "targetTitle"], ["postText", "targetTitle", "targetParagraphs"]]
    types = ["multiclass", "one_against_the_others"]


    # get confogurations
    configs = []
    for model in models:
        for fields in field_config:
            for type in types:
                if type == "multiclass":
                    mapping = {'passage': 0, 'phrase':1, 'multi':2}
                    configs.append((model, fields, mapping))


                elif type == "one_against_the_others":
                    classes = ["passage", "phrase", "multi"]
                    for class_ in classes:
                        mapping = {}
                        for c in classes:
                            if c == class_:
                                mapping[c] = 1
                            else:
                                mapping[c] = 0
                        configs.append((model, fields, mapping))

    for model, fields, mapping in configs:

        MODEL_NAME = config_name(model, fields, mapping)
        print("Start training")
        print("Configs:", model, fields, mapping)
        print("Name:", MODEL_NAME)

        # Load dataset
        dataset = load_dataset(fields=fields, files=data_paths, mapping=mapping)
        dataset = preprocess_dataset(dataset=dataset, model_base=MODEL_BASE, model_name=model)

        if model.endswith("full"):
            model = 


        # Logging
        if LOGGING:
            wandb.init(project=PROJECT_NAME, entity="jueri")


        # Train
        trainer_trained = create_model(MODEL_BASE, model, dataset)
        print(trainer_trained.evaluate())
        if LOGGING:
            wandb.finish()

   
        trainer_trained.save_model(os.path.join('./Models', MODEL_NAME+"-"+now))  # save model



# Models

### Fields
