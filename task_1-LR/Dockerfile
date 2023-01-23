FROM python:3.6

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update
RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'
	

COPY app.py /app.py	
COPY models/ /models/

RUN chmod +x /app.py /models/*
RUN /usr/local/bin/python3 -m spacy download en_core_web_lg

ENTRYPOINT ["/app.py" ]