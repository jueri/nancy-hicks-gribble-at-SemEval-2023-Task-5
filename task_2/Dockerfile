# Build: docker build -t registry.webis.de/code-research/tira/tira-user-nancy-hicks-gribble/task-2:0.0.1 .
# Run: docker run [image] --name task_2 --iniput Data/webis-clickbait-22/validation.json --output ./run.json
FROM python:3.6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py /app.py

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'

RUN chmod +x /app.py

ENTRYPOINT [ "python", "/app.py" ]
