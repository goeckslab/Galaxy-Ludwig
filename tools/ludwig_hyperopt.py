import logging
import os
import pickle
import sys

from jinja_report import generate_report

from ludwig.hyperopt_cli import cli
from ludwig.globals import (
    HYPEROPT_STATISTICS_FILE_NAME,
)
from ludwig.visualize import visualizations_registry

from model_unpickler import SafeUnpickler

import yaml


logging.basicConfig(level=logging.DEBUG)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])

# visualization
output_directory = None
for ix, arg in enumerate(sys.argv):
    if arg == "--output_directory":
        output_directory = sys.argv[ix+1]
        break

hyperopt_stats_path = os.path.join(
    output_directory,
    "hyperopt", HYPEROPT_STATISTICS_FILE_NAME
)

visualizations = ["hyperopt_report", "hyperopt_hiplot"]

viz_output_directory = os.path.join(output_directory, "visualizations")
for viz in visualizations:
    viz_func = visualizations_registry[viz]
    viz_func(
        hyperopt_stats_path=hyperopt_stats_path,
        output_directory=viz_output_directory,
        file_format="png",
    )

# report
title = "Ludwig Hyperopt"
report_config = {
    "title": title,
    "visualizations": [
        {
            "src": f"visualizations/{fl}",
            "type": "image" if fl[fl.rindex(".") + 1:] == "png" else
                    fl[fl.rindex(".") + 1:],
        } for fl in sorted(os.listdir(viz_output_directory))
    ],
    "raw stats": [
        {
            "src": os.path.basename(hyperopt_stats_path), "type": "json",
        },
    ],
}
with open(os.path.join(output_directory, "report_config.yml"), 'w') as fh:
    yaml.dump(report_config, fh)

report_path = os.path.join(output_directory, "smart_report.html")
with open(report_path, "w") as fh:
    html = generate_report.main(report_config, schema={"html_height": 800})
    fh.write(html)
