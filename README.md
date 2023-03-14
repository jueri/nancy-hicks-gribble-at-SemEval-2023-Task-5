# nancy-hicks-gribble-at-SemEval-2023-Task-5

This repository holds the code for the [SemEval 2023 Clickbait Spoiling](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html) shared task submission from the team `nancy-hicks-gribble`.

## Overview
The repository is structured as follows:
Data: is the directory where the scripts expect the [original dataset](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html#data) from the lab organizers.

- [Scripts](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/skripts): contains the code for pre-studies and data analysis. This is irrelevant for reproducing the contribution.

- [task_1-LR](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/task_1-LR): holds the code for the shallow learned models used in the comparison (not submitted)

- [task_1-multiclass](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/task_1-multiclass): contains the code for the submitted system `INJ-TASK1_MULTYCLASS`.

- [task_1-oao-base](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/task_1-oao-base): contains the code for the submitted system `INJ-TASK1_OAO`.

- [task_1-oao-news](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/task_1-oao-news): contains the code for the submitted system `INJ-TASK1_NEWS`.

- [task_2](https://github.com/jueri/nancy-hicks-gribble-at-SemEval-2023-Task-5/tree/main/task_2): contains the code for the single submission of the team for the second task. The system was called `SHORT-SCREW`.

## Models
Due to the size of the models used for submission and testing, they are not part of the repository but can be retrieved through [this link]().

|System|Container|Model|
|---|---|---|
|task_1-LR|--|`model.pkl`|
|task_1-LR|--|`preprocessor.pkl`|
|task_1-multiclass|`INJ-TASK1_MULTYCLASS`|roberta-base-3_multiclass-2022-12-21-T18-06-40|
|task_1-oao-base|`INJ-TASK1_OAO`|roberta-base-1_phrase-2022-12-21-T17-20-38|
|task_1-oao-base|`INJ-TASK1_OAO`|roberta-base-2_passage-2022-12-21-T17-34-28|
|task_1-oao-base|`INJ-TASK1_OAO`|roberta-base-3_multi-2022-12-21-T19-36-44|
|task_1-oao-news|`INJ-TASK1_NEWS`|roberta-news-full-1_phrase-2022-12-21-T20-36-22|
|task_1-oao-news|`INJ-TASK1_NEWS`|roberta-news-full-3_multi-2022-12-22-T11-54-08|
|task_1-oao-news|`INJ-TASK1_NEWS`|roberta-news-full-3_passage-2022-12-22-T10-50-31|
|task_2|`SHORT-SCREW`|[deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)|

## Reference
TBD