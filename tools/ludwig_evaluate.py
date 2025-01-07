import logging
import pickle
import sys

from ludwig.evaluate import cli

from ludwig_experiment import convert_parquet_to_csv, \
    generate_html_report, make_visualizations

from model_unpickler import SafeUnpickler


logging.basicConfig(level=logging.INFO)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])

ludwig_output_directory_name = ""

make_visualizations(ludwig_output_directory_name)

convert_parquet_to_csv(
    ludwig_output_directory_name
)

title = "Ludwig Evaluate"
generate_html_report(title, "")
