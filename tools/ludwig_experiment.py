import logging
import os
import pickle
import sys

from jinja_report import generate_report

from ludwig.experiment import cli
from ludwig.globals import (
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.visualize import get_visualizations_registry

from model_unpickler import SafeUnpickler

import yaml


logging.basicConfig(level=logging.DEBUG)

LOG = logging.getLogger(__name__)

setattr(pickle, 'Unpickler', SafeUnpickler)

# visualization
output_directory = None
for ix, arg in enumerate(sys.argv):
    if arg == "--output_directory":
        output_directory = sys.argv[ix+1]
        break

viz_output_directory = os.path.join(output_directory, "visualizations")


def make_visualizations(ludwig_output_directory_name):
    ludwig_output_directory = os.path.join(
        output_directory,
        ludwig_output_directory_name,
    )
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
        TEST_STATISTICS_FILE_NAME,
    )
    ground_truth_metadata = os.path.join(
        ludwig_output_directory,
        "model",
        TRAIN_SET_METADATA_FILE_NAME,
    )
    probabilities = os.path.join(
        ludwig_output_directory,
        PREDICTIONS_PARQUET_FILE_NAME,
    )

    for viz in visualizations:
        viz_func = get_visualizations_registry()[viz]
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
def render_report(
    title: str,
    ludwig_output_directory_name: str,
    show_visualization: bool = True
):
    ludwig_output_directory = os.path.join(
        output_directory,
        ludwig_output_directory_name,
    )
    report_config = {
        "title": title,
    }
    if show_visualization:
        report_config["visualizations"] = [
            {
                "src": f"visualizations/{fl}",
                "type": "image" if fl[fl.rindex(".") + 1:] == "png" else
                        fl[fl.rindex(".") + 1:],
            } for fl in sorted(os.listdir(viz_output_directory))
        ]
    report_config["raw outputs"] = [
        {
            "src": f"{fl}",
            "type": "json" if fl.endswith(".json") else "unclassified",
        } for fl in sorted(os.listdir(ludwig_output_directory))
        if fl.endswith((".json", ".parquet"))
    ]

    with open(os.path.join(output_directory, "report_config.yml"), 'w') as fh:
        yaml.dump(report_config, fh)

    report_path = os.path.join(output_directory, "smart_report.html")
    generate_report.main(
        report_config,
        schema={"html_height": 800},
        outfile=report_path,
    )


if __name__ == "__main__":

    cli(sys.argv[1:])

    ludwig_output_directory_name = "experiment_run"

    make_visualizations(ludwig_output_directory_name)
    title = "Ludwig Experiment"
    render_report(title, ludwig_output_directory_name)
