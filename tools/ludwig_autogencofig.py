import argparse

import logging

from ludwig import automl
from ludwig.utils import defaults

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Render a Ludwig config')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file', required=True)
    parser.add_argument('--output_feature', type=str, help='Name for the output feature', required=True)
    parser.add_argument('--output', type=str, help='Path for the output file', required=True)
    args = parser.parse_args()

    
    args_init = ["--dataset", args.dataset, "--target", args.output_feature, "--output", args.output]
    automl.cli_init_config(args_init)
    args_render = ["--config", args.output, "--output", args.output]
    defaults.cli_render_config(args_render)

if __name__ == "__main__":
    main()