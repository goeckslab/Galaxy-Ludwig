import logging
import pickle
import sys

from ludwig.train import cli

from ludwig_experiment import make_visualizations, render_report

from model_unpickler import SafeUnpickler


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])


make_visualizations()
title = "Ludwig Train"
render_report(title)
