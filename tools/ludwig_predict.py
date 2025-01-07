import logging
import pickle
import sys

from ludwig.predict import cli

from ludwig_experiment import convert_parquet_to_csv

from model_unpickler import SafeUnpickler


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])

convert_parquet_to_csv(
    ""
)
