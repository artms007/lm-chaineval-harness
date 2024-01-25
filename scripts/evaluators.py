from evaluate import load
import os
import re
from tqdm import tqdm
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# =====================
# Base Class
# =====================

#--metrics=pass@1,pass@k  --k=1

class Evaluator(object):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified metrics, and calculate scores.
    """

    def __init__(self, metric_id:str, args:dict, load_path:str = None):
        self.metric_id = metric_id # pass@1 pass@2
        self.eval = None if load_path is None else load(load_path)  # code_eval
        self.args = args

    def __repr__(self):
        return self.metric_id

    def score_item(self, item):
        item[self.metric_id] = 0.0

    def score(self, records):
        #scores = []
        score=0.0
        n = 0
        for record in tqdm(records, desc=f'Scoring {self.metric_id}={(score/max(n,1)):.3f}'):
            if self.metric_id not in record:
                self.score_item(record)
            score += record[self.metric_id]
            n+=1
        #     scores.append(record[self.metric_id])
        # if scores:
        #     total_score = sum(scores) / len(scores)
        # else:
        #     total_score = 0.0
        return {self.metric_id: score}

# HumanEval pass@1
#

def humaneval_extract(prompt, generated_text):
    # if generated_text == '':
    #     return 'Empty Code!!'
    stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
    min_stop_index = len(generated_text)
    for seq in stop_sequences:
        stop_index = generated_text.find(seq)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return prompt + "\n" + generated_text[:min_stop_index]


class CodeEvalEvaluator(Evaluator):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """

    def score_item(self, record):
        test_cases = [record['reference']]
        extracted_code = [humaneval_extract(record['model_input'], x) for x in record['extracted_results']]
        record['generated_code'] = extracted_code
        candidates = [extracted_code]
        pass_at_k, results = self.eval.compute(references=test_cases, predictions=candidates, k=[1])
        record['code_eval_results'] = results
        record[self.metric_id] = pass_at_k['pass@1']

class ExactMatchEvaluator(Evaluator):

    def score_item(self, data):
        predictions = data['model_output']
        references = data['reference']
        data['exact_match'] = self.metric.compute(predictions=predictions, references=references)['exact_match']
    

# 日本語用のtokenizer
# Python: 正規表現による簡易版形態素解析
# https://qiita.com/kinoshita_yuri/items/e15f143981f1616994ed
    
def tokenize_japaneses(text):
    pJA = re.compile(r"/|[A-Z]+|[a-z]+|[ァ-ンー]+|[ぁ-ん-]+|[ァ-ヶ]+|[一-龍]+|[。、]|/")
    text_m = []
    m = pJA.findall(text)
    for row in m:
        if re.compile(r'^[あ-ん]+$').fullmatch(row):
            if row[0] in 'はがのにへともでを':
                prefix = row[0]
                token = row[1:]
                text_m.append(prefix)
                if (len(token) > 0):
                    text_m.append(token)
            elif row[-2:] in 'のでからまで':
                token = row[0:-2]
                suffix = row[-2:]
                text_m.append(token)
                text_m.append(suffix)
            elif row[-1:] in 'もはがでを':
                token = row[0:-1]
                suffix = row[-1:]
                text_m.append(token)
                text_m.append(suffix)
            else:
                text_m.append(row)
        else:
            text_m.append(row)
    return text_m

class BLEUEvaluator(Evaluator):
    # def calculate(self, dataset, record):

    #     # BLEUメトリック用のデータ準備
    #     references = [[d['reference'].split()] for d in dataset]  # リストのリストとして分割された参照文
    #     candidates = [d['model_output'].split() for d in dataset]  # 分割された予測文のリスト
    #     # BLEU スコアを計算
    #     score = self.metric.compute(predictions=candidates, references=references)['bleu']

    #     for data in dataset:
    #         data['bleu_score'] = score

    #     return score, dataset

    
    def score_item(self, data):
        predictions = [data['extracted_result']]
        references = [[data['reference']]]
        if output_lang == 'ja':
            item_score = self.metric.compute(predictions=predictions, references=references, tokenier=tokenize_ja, smooth=True)['bleu']
        else:
            item_score = self.metric.compute(predictions=predictions, references=references, smooth=True)['bleu']
        self.item_scores.append(item_score)
        
        return item_score

    def total_calculate(self, dataset, record, output_lang):
        predictions = [data['formatted_output'] for data in dataset]
        references = [[data['reference']] for data in dataset]
        if output_lang == 'ja':
            total_score = self.metric.compute(predictions=predictions, references=references, tokenier=tokenize_ja, smooth=True)['bleu']
        else:
            total_score = self.metric.compute(predictions=predictions, references=references, smooth=True)['bleu']

        return total_score
        

class F1Evaluator(Evaluator):
    # def calculate(self, dataset, record):
        
    #     # F1スコアの計算に必要な正解ラベルと予測ラベルのリストを準備
    #     references = [d['reference'] for d in dataset]
    #     candidates = [d['model_output'] for d in dataset]
    #     # F1スコアを計算
    #     score = self.metric.compute(predictions=candidates, references=references)["f1"]
    #     # `score` には通常、precision, recall, f1 のキーが含まれている
    #     #f1_score = score['f1']
    #     #score = f1_score

    #     for data in dataset:
    #         data['f1_score'] = score

    #     return score, dataset
    def item_calculate(self, data, record, output_lang):
        return None
    
    def total_calculate(self, dataset, record, output_lang):
        predictions = [int(data['model_output']) for data in dataset]
        references = [int(data['reference']) for data in dataset]
        total_score = self.metric.compute(predictions=predictions, references=references)["f1"]
        return total_score

#######################

def load_evaluator(metric_id, args):
    if metric_id == "pass@1":
        return CodeEvalEvaluator("pass@1", args, load_path='code_eval')
    elif metric_id == "pass@k":
        k = args['pass_at_k|k|=1']
        return CodeEvalEvaluator(f"pass@{k}", args, load_path='code_eval')
    elif metric_id == "exact_match":
        return ExactMatchEvaluator("exact_match", args, load_path='exact_match')
    else:
        print(f"未定義の評価尺度//Unknown metrics: {metric_id}")
    
def compose_evaluators(args):
    metrics = args['metrics']
    evaluators = []
    if metrics is not None:
        for metric_id in metrics.split(','):
            eval = load_evaluator(metric_id.strip(), args)
            if eval:
                evaluators.append(eval)
    return evaluators