#!/bin/sh

virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl
pip install -r requirements.txt

cd wideep
exec python net_mk1.py "$@"

