import logging
import pickle
import sys

from model_unpickler import SafeUnpickler
from ludwig.hyperopt_cli import cli


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])
