FROM python:3.6
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update \
	&& apt-get install -y git-lfs

RUN wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/usr/local/lib/python3.6/site-packages/tira.py'
	

COPY app.py /app.py	
COPY models/ /models/

RUN chmod +x /app.py /models/*

ENTRYPOINT ["/app.py" ]