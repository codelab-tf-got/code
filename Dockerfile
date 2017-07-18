FROM tensorflow/tensorflow:latest

WORKDIR /code
ADD requirements.txt /code
RUN pip install -r requirements.txt 

ADD dataset /code/dataset
ADD wideep /code/wideep
ADD launch.sh /code
ADD setup.py /code

CMD ["python", "./wideep/net_mk1.py"]