import json
import sys
import yaml
from ludwig.utils.defaults import render_config


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

output = sys.argv[2]
render_config(config, output)

with open('./pre_config.yml', 'w') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)