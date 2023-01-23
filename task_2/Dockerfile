# Build: docker build -t registry.webis.de/code-research/tira/tira-user-nancy-hicks-gribble/task-1-oao-news:0.0.1 .
# Run: docker run [image] --input Data/webis-clickbait-22/validation.jsonl --output ./run.jsonl
FROM python:3.6

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py /app.py

RUN apt-get update \
	&& apt-get install -y git-lfs wget \
	&& wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py' \
	&& git clone 'https://huggingface.co/deepset/roberta-base-squad2' /model \
	&& cd /model \
	&& git lfs install \
	&& git fetch \
	&& rm -Rf .git


RUN chmod +x /app.py

# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["/app.py" ]
