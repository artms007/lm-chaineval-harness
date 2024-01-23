import argparse
import json
import time
import yaml

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config YAML file')
    parser.add_argument('--model_path', type=str, help='Path to the model file')
    parser.add_argument('--model_args', type=json.loads, default=None, help='Model arguments in JSON format')
    parser.add_argument('--quantize_model', action='store_true', help='Enable model quantization with bitsandbytes')
    parser.add_argument('--openai_api_key', type=str, default=None, help='OpenAI API token')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token')
    parser.add_argument('--aws_access_key_id', type=str, default=None, help='AWS access key ID')
    parser.add_argument('--aws_secret_access_key', type=str, default=None, help='AWS secret access key')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset file')
    parser.add_argument('--dataset_args', type=json.loads, default=None, help='Dataset arguments in JSON format')
    parser.add_argument('--template_path', type=str, help='Path to the template file')
    parser.add_argument('--metric_path', type=str, default=None, help='Path to the metric file')
    parser.add_argument('--metric_args', type=json.loads, default=None, help='Metric arguments in JSON format')
    parser.add_argument('--result_path', type=str, default=None, help='Path to the result file')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for verbose output')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        for section, settings in config.items():
            if settings is not None:
                for setting, value in settings.items():
                    arg_name = setting
                    if hasattr(args, arg_name):
                        if getattr(args, arg_name) is None:
                            setattr(args, arg_name, value)

    check_required_args(args)

    if args.result_path is None:
        args.result_path = f'./logs/result_{int(time.time())}.jsonl'

    return args

def check_required_args(args):
    required_args = ['model_path', 'dataset_path', 'template_path']
    for arg in required_args:
        if getattr(args, arg, None) is None:
            raise ValueError(f"Error: '{arg}' is required but not provided in command line arguments or config file.")


import sys

def _parse_value(value):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value

def _parse_key_value(arg, next_value):
    if '=' in arg:
        if key.startswith('--'):
            key = key[2:]
        key, _, value = arg.partition('=')
        return key, _parse_value(value)     
    elif key.startswith('--'):
        key = key[2:]
        if next_value.startswith('--'):
            if key.startswith('enable_'):
                return key[7:], True
            elif key.startswith('disable_'):
                return key[8:], False
            return key, True
        else:
            return key, _parse_value(next_value)
    else:
        return 'files', key
    
def adhoc_argument_parser():
    argv = sys.argv[1:]
    args={}
    for arg, next_value in zip(argv, argv[1:] + ['']):
        key, value = _parse_key_value(arg, next_value)
        if value:
            args[key] = value
    return args