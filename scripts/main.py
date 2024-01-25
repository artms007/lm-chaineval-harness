from typing import List
import json
import os
from tqdm import tqdm
from models import load_model
from dataloaders import load_evaldata
from templates import load_template
from evaluators import compose_evaluators
from adhoc import adhoc_argument_parser


def guess_uniquekey(dataset: List[dict]):
    for key in dataset[0].keys():
        if 'id' in key.lower():
            return key
    return None

def new_records(dataset):
    keyid = guess_uniquekey(dataset)
    if keyid:
        return [{'unique_id': data[keyid]} for data in dataset]
    else:
        return [{'unique_id': f'index/{n}'} for n in range(len(dataset))]

def load_records(result_path, dataset):
    """Load existing results from the file."""
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return new_records(dataset)

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

    dataset_id, dataset = load_evaldata(args)

    template = load_template(args, dataset)

    result_path = args['result_path|record_path']
    if result_path and args['resume|=false']:
        records = load_records(result_path, dataset)
    else:
        records = new_records(dataset) 

    model = load_model(args)
    if model:
        test_run = args['test_run|=false']
        if result_path is None:
            model_id = (f'{model}').replace('/', '_')
            result_path = f'{dataset_id}_{model_id}.jsonl'
            args.verbose_print(f'保存先//Saving.. {result_path}')
        
        n = args['num_return_sequences|n|N|=1']
        args.verbose_print(f"モデル評価//Text-generation: {model} n={n}")

        if test_run:
            args.verbose_print('テスト実行のため先頭5件のみ実行します')
            result_path = result_path.replace('.json', '_test_run.json')
            records = records[:5]
        
        for i, record in enumerate(tqdm(records, desc=f'Inferencing {model}')):
            source = dataset[i]
            if 'model_input' not in record:
                record['model_input'] = template.create_prompt(source)
            if 'reference' not in record:
                record['reference'] = template.create_reference(source)
            if 'model_outputs' not in record:
                record['model_outputs'] = model.generate_list(record['model_input'], n=n)
                record['model_output'] = record['model_outputs'][0]
            if 'extracted_results' not in record:
                record['extracted_results'] = template.extract(record['model_outputs'])
                record['extracted_result'] = record['extracted_results'][0]
            save_records(result_path, records)

    evaluators = compose_evaluators(args)
    if len(evaluators) > 0 and result_path:
        args.verbose_print(f"評価尺度//Metrics: {evaluators}")
        results = {}
        for eval in evaluators:
            results.update(eval.score(records))
            save_records(result_path, records)
        print(f"スコア//Scores: {results}")
        args.update({'score': results})

    if result_path:
        save_records(result_path, records, args)
    
    args.utils_check()

if __name__ == '__main__':
    main()
