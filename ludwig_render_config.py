import json
import sys


inputs = sys.argv[1]
with open(inputs, 'r') as handler:
    params = json.load(handler)

