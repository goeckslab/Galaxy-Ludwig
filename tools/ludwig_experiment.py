import logging
import os
import pickle
import sys

from jinja_report import generate_report

from ludwig.experiment import cli
from ludwig.visualize import visualizations_registry

from model_unpickler import SafeUnpickler

import yaml


logging.basicConfig(level=logging.DEBUG)

LOG = logging.getLogger(__name__)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])

# visualization
output_directory = None
for ix, arg in enumerate(sys.argv):
    if arg == "--output_directory":
        output_directory = sys.argv[ix+1]
        break

ludwig_output_directory = os.path.join(output_directory, "experiment_run")
viz_output_directory = os.path.join(output_directory, "visualizations")


def make_visualizations():
    visualizations = [
        "confidence_thresholding",
        "confidence_thresholding_data_vs_acc",
        "confidence_thresholding_data_vs_acc_subset",
        "confidence_thresholding_data_vs_acc_subset_per_class",
        "confidence_thresholding_2thresholds_2d",
        "confidence_thresholding_2thresholds_3d",
        "binary_threshold_vs_metric",
        "roc_curves",
        "roc_curves_from_test_statistics",
        "calibration_1_vs_all",
        "calibration_multiclass",
        "confusion_matrix",
        "frequency_vs_f1",
        "learning_curves",
    ]

    training_statistics = os.path.join(
        ludwig_output_directory,
        "training_statistics.json",
    )
    test_statistics = os.path.join(
        ludwig_output_directory,
        "test_statistics.json",
    )
    ground_truth_metadata = os.path.join(
        ludwig_output_directory,
        "model",
        "training_set_metadata.json",
    )
    probabilities = os.path.join(
        ludwig_output_directory,
        "predictions.parquet",
    )

    for viz in visualizations:
        viz_func = visualizations_registry[viz]
        try:
            viz_func(
                training_statistics=[training_statistics],
                test_statistics=[test_statistics],
                probabilities=[probabilities],
                top_n_classes=[0],
                output_feature_name=[],
                ground_truth_split=2,
                top_k=3,
                ground_truth_metadata=ground_truth_metadata,
                output_directory=viz_output_directory,
                normalize=False,
                file_format="png",
            )
        except Exception as e:
            LOG.info(e)


# report
def render_report(title):
    report_config = {
        "title": title,
        "visualizations": [
            {
                "src": f"visualizations/{fl}",
                "type": "image" if fl[fl.rindex(".") + 1:] == "png" else
                        fl[fl.rindex(".") + 1:],
            } for fl in sorted(os.listdir(viz_output_directory))
        ],
        "raw outputs": [
            {
                "src": f"{fl}",
                "type": "json" if fl.endswith(".json") else "unclassified",
            } for fl in sorted(os.listdir(ludwig_output_directory))
            if fl.endswith((".json", ".parquet"))
        ],
    }
    with open(os.path.join(output_directory, "report_config.yml"), 'w') as fh:
        yaml.dump(report_config, fh)

    report_path = os.path.join(output_directory, "smart_report.html")
    with open(report_path, "w") as fh:
        html = generate_report.main(report_config, schema={"html_height": 800})
        fh.write(html)


make_visualizations()
title = "Ludwig Experiment"
render_report(title)
