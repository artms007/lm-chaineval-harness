import os
import json
from collections import defaultdict


def load_existing_results(result_path, n):
    """Load existing results from the file."""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return [{} for _ in range(n)]


def group_and_aggregate_results(results):
    grouped_results = defaultdict(lambda: defaultdict(list))
    for result in results:
        id_value = result.get('id')
        if id_value:
            for key, value in result.items():
                if key in ['model_output', 'formatted_output']:
                    grouped_results[id_value][key].append(value)
                else:
                    grouped_results[id_value][key] = value
    return [dict(result) for result in grouped_results.values()]


def find_id_value(data):
    for key in data.keys():
        if 'id' in key:
            return data[key]
    return None


def find_unprocessed_data(dataset, existing_results):
    """Find data in the dataset that has not been processed yet."""
    processed_ids = {find_id_value(result) for result in existing_results}
    return [data for data in dataset if find_id_value(data) not in processed_ids]


def save_results(result_path, dataset, record, total_score=-1):
    """Save the evaluation results to a file."""
    mode = 'a' if total_score == -1 else 'w'
    
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(result_path, mode, encoding='utf-8') as f:
        for data in dataset:
            filtered_data = record.copy()
            for k, v in data.items():
                if 'id' in k:
                    filtered_data['id'] = v
            filtered_data['model_input'] = data.get('model_input', '')
            model_outputs = data.get('model_output', [])
            formatted_outputs = data.get('formatted_output', [])
            format_checked = data.get('format_checked', [])
            
            for i, (model_output, formatted_output, format_check) in enumerate(zip(model_outputs, formatted_outputs, format_checked)):
                filtered_data['model_output'] = model_output
                filtered_data['formatted_output'] = formatted_output

                filtered_data['format_checked'] = format_check if format_checked else ''

                filtered_data['output_format'] = data.get('output_format', '')
                filtered_data['reference'] = data.get('reference', '')
                filtered_data['item_score'] = data.get('item_score', '')
                filtered_data['total_score'] = total_score

                f.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
