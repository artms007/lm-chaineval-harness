import os
import time
import yaml
import argparse
import json
import argparse
from tqdm import tqdm
from config_utils import parse_args_and_config, load_config
from results_handling import load_existing_results, group_and_aggregate_results, find_id_value, find_unprocessed_data, save_results
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import load_evaluator

debug_mode = False

def debug_print(*messages):
    """
    debug_mode がTrue のとき、デバッグ出力する
    """
    global debug_mode
    if debug_mode:
        print("🐥", *messages)
        

def main():
    global debug_mode

    args = parse_args_and_config()
    debug_mode = args.debug_mode
    quantize = args.quantize_model
    debug_print("Quantization:\n", quantize)

    model = load_model(args.model_path, args.openai_api_key, args.aws_access_key_id, args.aws_secret_access_key, args.hf_token, args.model_args, quantize)
    debug_print("Model loaded:\n", model)

    dataset = load_testdata(args.dataset_path, args.dataset_args)
    debug_print(args.debug_mode, "Dataset loaded:\n", len(dataset), "entries")

    template = load_template(args.template_path)
    evaluator = load_evaluator(args.metric_path, args.metric_args)

    record = {
        'model': args.model_path,
        'dataset': args.dataset_path,
        'template': args.template_path,
        'metrics': args.metric_path,
    }

    loaded_results = load_existing_results(args.result_path)
    existing_results = group_and_aggregate_results(loaded_results)
    unprocessed_data = find_unprocessed_data(dataset, existing_results)

    for data in tqdm(unprocessed_data):
            
        prompt = template.process(data)
        data['reference'] = template.process_reference(data)
        data['model_input'] = prompt
        debug_print("Input:\n", data['model_input'])
        data['model_output'] = model.generate(prompt)
        debug_print("Output_Sample:\n", data['model_output'][0])
        output_lang, output_format, formatted_output_list, format_checked_list = template.collate(prompt, data['model_output'])
        data['output_format'] = output_format

        if format_checked_list:
            data['format_checked'] = format_checked_list
        
        data['formatted_output'] = formatted_output_list
        debug_print("Formatted_Sample:\n", data['formatted_output'][0])

        if evaluator:
            if data['formatted_output'] is None:
                data['item_score'] = 0.0
            else:
                data['item_score'] = evaluator.item_calculate(data, record, output_lang)
                debug_print("Score:\n", data['item_score'])

        save_results(args.result_path, [data], record)
    
    if evaluator:
        if 'output_lang' not in locals():
            raise ValueError("output_lang is not defined. Cannot execute as all data has already been processed.")

        all_data = existing_results + unprocessed_data
        total_score = evaluator.total_calculate(all_data, record, output_lang)
        save_results(args.result_path, all_data, record, total_score)
        print("Total_score:\n", total_score)

if __name__ == '__main__':
    main()
