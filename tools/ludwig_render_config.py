import json
import sys
import yaml
from ludwig.utils.defaults import merge_with_defaults


inputs = sys.argv[1]
with open(inputs, 'r') as handler:
    params = json.load(handler)

config = {}
# input features
config['input_features'] = []
for ftr in params['input_features']['input_feature']:
    config['input_features'].append(ftr['input_feature_selector'])
 
# output features
config['output_features'] = []
for ftr in params['output_features']['output_feature']:
    config['output_features'].append(ftr['output_feature_selector'])

# combiner
config['combiner'] = params['combiner']

# training
config["trainer"] = params["trainer"]["trainer"]
config['model_type'] = config['trainer'].pop("model_type")

# hyperopt
if params['hyperopt']['do_hyperopt']:
    config['hyperopt'] = params['hyperopt']['hyperopt']

with open('./pre_config.yml', 'w') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

output = sys.argv[2]
output_config = merge_with_defaults(config)

def clean_proc_column (config: dict) -> None:
    for ftr in config["input_features"]:
        ftr.pop("proc_column", None)
    for ftr in config["output_features"]:
        ftr.pop("proc_column", None)

clean_proc_column(output_config)

with open(output, "w") as f:
    yaml.safe_dump(output_config, f, sort_keys=False)