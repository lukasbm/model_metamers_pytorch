import argparse
import os
import pickle
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-file", metavar="-f", type=str)
args = parser.parse_args()

path = os.path.abspath(args.file)

with open(path, "rb") as file:
    x = pickle.load(file)
    pprint(x)
