import json
import logging
import sys

from ludwig.constants import (
    COMBINER,
    HYPEROPT,
    INPUT_FEATURES,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PROC_COLUMN,
    TRAINER,
)
from ludwig.schema.model_types.utils import merge_with_defaults

import yaml

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)
inputs = sys.argv[1]
with open(inputs, 'r') as handler:
    params = json.load(handler)

config = {}
# input features
config[INPUT_FEATURES] = []
for ftr in params[INPUT_FEATURES]['input_feature']:
    config[INPUT_FEATURES].append(ftr['input_feature_selector'])

# output features
config[OUTPUT_FEATURES] = []
for ftr in params[OUTPUT_FEATURES]['output_feature']:
    config[OUTPUT_FEATURES].append(ftr['output_feature_selector'])

# combiner
config[COMBINER] = params[COMBINER]

# training
config[TRAINER] = params[TRAINER][TRAINER]
config[MODEL_TYPE] = config[TRAINER].pop(MODEL_TYPE)

# hyperopt
if params[HYPEROPT]['do_hyperopt'] == 'true':
    config[HYPEROPT] = params[HYPEROPT][HYPEROPT]

with open('./pre_config.yml', 'w') as f:
    yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)

output = sys.argv[2]
output_config = merge_with_defaults(config)


def clean_proc_column(config: dict) -> None:
    for ftr in config[INPUT_FEATURES]:
        ftr.pop(PROC_COLUMN, None)
    for ftr in config[OUTPUT_FEATURES]:
        ftr.pop(PROC_COLUMN, None)


clean_proc_column(output_config)

with open(output, "w") as f:
    yaml.safe_dump(output_config, f, sort_keys=False)
