#!/bin/sh

virtualenv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd wideep
exec python net_mk1.py "$@"

