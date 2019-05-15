FROM python:3.4-alpine
ADD . /trab
WORKDIR /trab
RUN pip3 install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('all')" ]
CMD ["python", "app.py"]
