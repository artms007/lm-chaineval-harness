from typing import List
import json
import os
from tqdm import tqdm
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import compose_evaluators
from adhoc import adhoc_argument_parser


def guess_uniquekey(dataset: List[dict]):
    for key in dataset[0].keys():
        if 'id' in key.lower():
            return key
    return None

def load_records(result_path, dataset):
    """Load existing results from the file."""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        keyid = guess_uniquekey(dataset)
        if keyid:
            return [{'unique_id': data[keyid]} for data in dataset]
        else:
            return [{'unique_id': f'index/{n}'} for n in range(len(dataset))]

def save_records(result_path, records, args=None):
    directory = os.path.dirname(result_path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(result_path, 'w', encoding='utf-8') as w:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=w)

    if args:
        savefile = result_path.replace('.jsonl', '_config.json')
        args.save_as_json(savefile)

def main():
    args = adhoc_argument_parser()

    dataset = load_testdata(args)
    args.verbose_print(f"Dataset loaded: {len(dataset)} entries")

    template = load_template(args)

    result_path = args['result_path|record_path']
    if result_path is None:
        result_path = 'dummy.jsonl'
    records = load_records(result_path, dataset)

    model = load_model(args)
    if model:
        n = args['num_return_sequences|n|N|=1']
        args.verbose_print(f"Model loaded: {model} n={n}")
        for i, record in enumerate(tqdm(records, desc=f'Inferencing {model}')):
            source = dataset[i]
            if 'model_input' not in record:
                record['model_input'] = template.create_prompt(source)
            if 'reference' not in record:
                record['reference'] = template.create_reference(source)
            if n == 1:
                if 'model_output' not in record:
                    record['model_output'] = model.generate(record['model_input'], n=1)
                    record['model_outputs'] = [record['model_output']]
            else:
                if 'model_outputs' not in record:
                    record['model_outputs'] = model.generate(record['model_input'], n=n)
                    record['model_output'] = record['model_outputs'][0]
            if 'extracted_result' not in record:
                record['extracted_result'] = template.extract(record['model_output'])
            if 'extracted_results' not in record:
                record['extracted_results'] = template.extract(record['model_outputs'])
            save_records(result_path, records)

    evaluators = compose_evaluators(args)
    if len(evaluators) > 0:
        args.verbose_print(f"Metrics: {evaluators}")
        results = {}
        for eval in evaluators:
            results.update(eval.score(records))
            save_records(result_path, records)
        print(f"Scores: {results}")
    save_records(result_path, records, args)

if __name__ == '__main__':
    main()
