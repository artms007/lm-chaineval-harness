import json
from datasets import load_dataset

def load_testdata(dataset_path: str, args):
    dataset = []
    n = 10
    for i in range(1, n + 1):
        dataset.append({
            "task_id": f"test_{i}",
            "prompt": f"test_prompt_{i}",
            "canonical_solution": f"test_solution_{i}",
            "test": f"test_test_{i}",
            "entry_point": f"test_entry_{i}"
        })
    args['_dataset_id'] = 'dummy/testdata'
    return dataset

def load_jsonl(dataset_path:str, args):
    dataset = []
    try:
        with open(dataset_path, 'r') as f:
            dataset = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {dataset_path} does not exist.")
    if '/' in dataset_path:
        _, _, dataset_path = dataset_path.rpartition('/')
    args['_dataset_id'] = dataset_path.replace('.jsonl', '')
    return dataset

def load_hfdataset(dataset_path:str, args):
    subargs = args.subset(prefix='dataset_')
    if 'split' not in subargs:
        subargs['split'] = args['split|=test']
    dataset = load_dataset(dataset_path, **subargs)
    args.verbose_print(dataset_path, dataset)
    dataset = [{k: v for k, v in item.items()} for item in dataset]
    if '/' in dataset_path:
        _, _, dataset_path = dataset_path.rpartition('/')
    if 'name' in subargs:
        name = subargs['name']
        dataset_path = f'{dataset_path}_{name}'
    args['_dataset_id'] = dataset_path
    return dataset

def load_dict(args):
    dataset_path = args['dataset|evaldata']
    if dataset_path is None:
        return load_testdata('dummy_testdata', args)
    elif dataset_path.endswith(".jsonl"):
        return load_jsonl(dataset_path, args)
    else:
        return load_hfdataset(dataset_path, args)

def load_evaldata(args):
    dataset_path = args['dataset|evaldata']
    dataset = load_dict(args)
    dataset_id = args['_dataset_id']
    dumpdata = json.dumps(dataset[0], indent=4, ensure_ascii=False)
    args.verbose_print(f'データセットの確認 {dataset_path}[{dataset_id}] {len(dataset)} entries\n{dumpdata}')
    return dataset
