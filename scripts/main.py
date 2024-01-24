import argparse
from tqdm import tqdm
from config_utils import parse_args_and_config, load_config
from results_handling import load_existing_results, group_and_aggregate_results, find_id_value, find_unprocessed_data, save_results
from models import load_model
from dataloaders import load_testdata
from templates import load_template
from evaluators import compose_evaluators
from scripts.adhoc import adhoc_argument_parser

def main():
    args = adhoc_argument_parser()

    dataset = load_testdata(args)
    args.verbose_print(f"Dataset loaded: {len(dataset)} entries")

    template = load_template(args.template_path)

    result_path = args['result_path|record_path']
    records = load_existing_results(result_path, len(dataset))
    # existing_results = group_and_aggregate_results(loaded_results)
    # unprocessed_data = find_unprocessed_data(dataset, existing_results)

    model = load_model(args)
    args.verbose_print(f"Model loaded: {model}")

    if model:
        n = args['num_return_sequences|n|N|=1']
        for i, record in enumerate(tqdm(records)):
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
            save_results(result_path, records)

    evaluators = compose_evaluators(args)
    if len(evaluators) > 0:
        args.verbose_print(f"Metrics: {evaluators}")
        results = {}
        for eval in evaluators:
            results.update(eval.score(records))
            save_results(result_path, records)
        print(f"Total_score: {results}")

if __name__ == '__main__':
    main()
