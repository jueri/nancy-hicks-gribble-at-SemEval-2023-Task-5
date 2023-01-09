FROM python:3.6

RUN pip3 install pandas jupyterlab docker transformers tqdm

COPY app.py /app.py

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'

ENTRYPOINT [ "/app.py" ]