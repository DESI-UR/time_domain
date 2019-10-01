#!/usr/bin/env python
from keras.models import load_model
from argparse import ArgumentParser

parser = ArgumentParser(description='CNN Loader')
parser.add_argument('--path', required=True,\
  help="Directory of Model")
args = parser.parse_args()

load_model(args.path)
