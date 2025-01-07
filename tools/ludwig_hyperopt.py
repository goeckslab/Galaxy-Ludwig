import logging
import os
import pickle
import sys

from ludwig.globals import (
    HYPEROPT_STATISTICS_FILE_NAME,
)
from ludwig.hyperopt_cli import cli
from ludwig.visualize import get_visualizations_registry

from model_unpickler import SafeUnpickler

from utils import (
    encode_image_to_base64,
    get_html_closing,
    get_html_template
)

logging.basicConfig(level=logging.DEBUG)

LOG = logging.getLogger(__name__)

setattr(pickle, 'Unpickler', SafeUnpickler)

cli(sys.argv[1:])


def generate_html_report(title):

    # Read test statistics JSON and convert to HTML table
    # try:
    #     test_statistics_path = hyperopt_stats_path
    #     with open(test_statistics_path, "r") as f:
    #         test_statistics = json.load(f)
    #     test_statistics_html = "<h2>Hyperopt Statistics</h2>"
    #     test_statistics_html += json_to_html_table(test_statistics)
    # except Exception as e:
    #     LOG.info(f"Error reading hyperopt statistics: {e}")

    plots_html = ""
    # Convert visualizations to HTML
    hyperopt_hiplot_path = os.path.join(
        viz_output_directory, "hyperopt_hiplot.html")
    if os.path.isfile(hyperopt_hiplot_path):
        with open(hyperopt_hiplot_path, "r", encoding="utf-8") as file:
            hyperopt_hiplot_html = file.read()
            plots_html += f'<div class="hiplot">{hyperopt_hiplot_html}</div>'

    # Iterate through other files in viz_output_directory
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
        <h2>Visualizations</h2>
        {plots_html}
    {get_html_closing()}
    """

    # Save the HTML report
    report_name = title.lower().replace(" ", "_")
    report_path = os.path.join(output_directory, f"{report_name}_report.html")
    with open(report_path, "w") as report_file:
        report_file.write(html_content)

    LOG.info(f"HTML report generated at: {report_path}")


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
    viz_func = get_visualizations_registry()[viz]
    viz_func(
        hyperopt_stats_path=hyperopt_stats_path,
        output_directory=viz_output_directory,
        file_format="png",
    )

# report
title = "Ludwig Hyperopt"
generate_html_report(title)
