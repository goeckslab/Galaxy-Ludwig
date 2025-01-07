import json
import logging
import os
import pickle
import sys

from jinja_report import generate_report

from ludwig.experiment import cli
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME
)
from ludwig.utils.data_utils import get_split_path
from ludwig.visualize import get_visualizations_registry

from model_unpickler import SafeUnpickler

import pandas as pd

from utils import (
    encode_image_to_base64,
    get_html_closing,
    get_html_template
)

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


def get_output_feature_name(experiment_dir, output_feature=0):
    """Helper function to extract specified output feature name.

    :param experiment_dir: Path to the experiment directory
    :param output_feature: position of the output feature the description.json
    :return output_feature_name: name of the first output feature name
                        from the experiment
    """
    if os.path.exists(os.path.join(experiment_dir, DESCRIPTION_FILE_NAME)):
        description_file = os.path.join(experiment_dir, DESCRIPTION_FILE_NAME)
        with open(description_file, "rb") as f:
            content = json.load(f)
        output_feature_name = \
            content["config"]["output_features"][output_feature]["name"]
        dataset_path = content["dataset"]
        return output_feature_name, dataset_path
    return None, None


def check_file(file_path):
    """Check if the file exists; return None if it doesn't."""
    return file_path if os.path.exists(file_path) else None


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

    # Check existence of required files
    training_statistics = check_file(os.path.join(
        ludwig_output_directory,
        "training_statistics.json",
    ))
    test_statistics = check_file(os.path.join(
        ludwig_output_directory,
        TEST_STATISTICS_FILE_NAME,
    ))
    ground_truth_metadata = check_file(os.path.join(
        ludwig_output_directory,
        "model",
        TRAIN_SET_METADATA_FILE_NAME,
    ))
    probabilities = check_file(os.path.join(
        ludwig_output_directory,
        PREDICTIONS_PARQUET_FILE_NAME,
    ))

    output_feature, dataset_path = get_output_feature_name(
        ludwig_output_directory)
    ground_truth = None
    split_file = None
    if dataset_path:
        ground_truth = check_file(dataset_path)
        split_file = check_file(get_split_path(dataset_path))

    if (not output_feature) and (test_statistics):
        test_stat = os.path.join(test_statistics)
        with open(test_stat, "rb") as f:
            content = json.load(f)
        output_feature = next(iter(content.keys()))

    for viz in visualizations:
        viz_func = get_visualizations_registry()[viz]
        try:
            viz_func(
                training_statistics=[training_statistics]
                if training_statistics else [],
                test_statistics=[test_statistics] if test_statistics else [],
                probabilities=[probabilities] if probabilities else [],
                top_n_classes=[0],
                output_feature_name=output_feature if output_feature else "",
                ground_truth_split=2,
                top_k=3,
                ground_truth_metadata=ground_truth_metadata,
                ground_truth=ground_truth,
                split_file=split_file,
                output_directory=viz_output_directory,
                normalize=False,
                file_format="png",
            )
        except Exception as e:
            LOG.info(f"Visualization: {viz}")
            LOG.info(f"Error: {e}")


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
        yaml.safe_dump(report_config, fh)

    report_path = os.path.join(output_directory, "smart_report.html")
    generate_report.main(
        report_config,
        schema={"html_height": 800},
        outfile=report_path,
    )


def convert_parquet_to_csv(ludwig_output_directory_name):
    """Convert the predictions Parquet file to CSV."""
    ludwig_output_directory = os.path.join(
        output_directory, ludwig_output_directory_name)
    parquet_path = os.path.join(
        ludwig_output_directory, "predictions.parquet")
    csv_path = os.path.join(
        ludwig_output_directory, "predictions_parquet.csv")

    try:
        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
        LOG.info(f"Converted Parquet to CSV: {csv_path}")
    except Exception as e:
        LOG.error(f"Error converting Parquet to CSV: {e}")


def generate_html_report(title, ludwig_output_directory_name):
    # ludwig_output_directory = os.path.join(
    #     output_directory, ludwig_output_directory_name)

    # test_statistics_html = ""
    # # Read test statistics JSON and convert to HTML table
    # try:
    #     test_statistics_path = os.path.join(
    #         ludwig_output_directory, TEST_STATISTICS_FILE_NAME)
    #     with open(test_statistics_path, "r") as f:
    #         test_statistics = json.load(f)
    #     test_statistics_html = "<h2>Test Statistics</h2>"
    #     test_statistics_html += json_to_html_table(
    #         test_statistics)
    # except Exception as e:
    #     LOG.info(f"Error reading test statistics: {e}")

    # Convert visualizations to HTML
    plots_html = ""
    if len(os.listdir(viz_output_directory)) > 0:
        plots_html = "<h2>Visualizations</h2>"
    for plot_file in sorted(os.listdir(viz_output_directory)):
        plot_path = os.path.join(viz_output_directory, plot_file)
        if os.path.isfile(plot_path) and plot_file.endswith((".png", ".jpg")):
            encoded_image = encode_image_to_base64(plot_path)
            plots_html += (
                f'<div class="plot">'
                f'<h3>{os.path.splitext(plot_file)[0]}</h3>'
                '<img src="data:image/png;base64,'
                f'{encoded_image}" alt="{plot_file}">'
                f'</div>'
            )

    # Generate the full HTML content
    html_content = f"""
    {get_html_template()}
        <h1>{title}</h1>
        {plots_html}
    {get_html_closing()}
    """

    # Save the HTML report
    title: str
    report_name = title.lower().replace(" ", "_")
    report_path = os.path.join(output_directory, f"{report_name}_report.html")
    with open(report_path, "w") as report_file:
        report_file.write(html_content)

    LOG.info(f"HTML report generated at: {report_path}")


if __name__ == "__main__":

    cli(sys.argv[1:])

    ludwig_output_directory_name = "experiment_run"

    make_visualizations(ludwig_output_directory_name)
    # title = "Ludwig Experiment"
    # render_report(title, ludwig_output_directory_name)
    convert_parquet_to_csv(ludwig_output_directory_name)
    generate_html_report("Ludwig Experiment", ludwig_output_directory_name)
