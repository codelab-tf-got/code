FROM tensorflow/tensorflow:latest-py3

WORKDIR /code
ADD requirements.txt /code
RUN pip3 install -r requirements.txt 

ADD dataset /code/dataset
ADD wideep /code/wideep
ADD launch.sh /code
ADD setup.py /code

CMD ["python", "./wideep/net_mk1.py"]