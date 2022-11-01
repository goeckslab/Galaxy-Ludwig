import logging
import pickle
import sys

from ludwig.predict import cli

from ludwig_experiment import render_report

from model_unpickler import SafeUnpickler


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])

ludwig_output_directory_name = ""
title = "Ludwig Prediction"
render_report(title, ludwig_output_directory_name, show_visualization=False)
