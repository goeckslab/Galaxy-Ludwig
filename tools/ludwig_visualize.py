import logging
import pickle
import sys

from ludwig.visualize import cli

from model_unpickler import SafeUnpickler


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])
