from evaluate import load
import os
import re
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# =====================
# Base Class
# =====================

class Evaluator(object):
    """
    Base class for evaluators that use a model to obtain answers for generated prompts,
    evaluate them based on specified metrics, and calculate scores.
    """

    def __init__(self, metric_id:str, eval_path:str, args:dict):
        self.metric_id = metric_id
        self.eval = None if eval_path is None else load(eval_path) 
        self.args = args

    def __repr__(self):
        return self.metric_id

    def score_item(self, item):
        item[self.metric_id] = 0.0

    def score(self, records):
        scores = []
        for item in records:
            if self.metric_id not in item:
                self.score_item(item)
            scores.append(item[self.metric_id])
        if scores:
            total_score = sum(scores) / len(scores)
        else:
            total_score = 0.0
        return {self.metric_id: total_score}


class CodeEvalEvaluator(Evaluator):
    """
    コード評価用Evaluatorクラス。HuggingFaceのevaluate-metric/code_evalを使用してスコアを算出する。
    """

    def is_blank(self, candidates):
        """
        何をやっているのか？
        """
        if isinstance(candidates, list):
            for sublist in candidates:
                if not isinstance(sublist, list) or not all(isinstance(item, str) and item.strip() == '' for item in sublist):
                    return False
            return True
        return False

    def score_item(self, data):
        test_cases = [data['reference']]
        candidates = [data['formatted_output']]
        if not self.is_blank(candidates):
            pass_at_k, results = self.eval.compute(references=test_cases, predictions=candidates, k=[1])
            data['code_eval'] = results
            data[self.metric_id] = pass_at_k['pass@1']
        else:
            data[self.metric_id] = 0.0

class ExactMatchEvaluator(Evaluator):

    def score_item(self, data):
        predictions = data['model_output']
        references = data['reference']
        data['exact_match'] = self.metric.compute(predictions=predictions, references=references)['exact_match']
    

# =====================
# Testing Code
# =====================

class TestEvaluator(Evaluator):
    """
    テスト用Evaluatorクラス。単に文字列の長さでスコアを算出する。
    """

    # def calculate(self, dataset, record):
    #     # データセット内のすべてのoutputの長さの平均を計算してスコアとする
    #     total_length = sum(len(data['formatted_output']) for data in dataset)
    #     result_score = total_length / len(dataset) if dataset else 0
    #     for data in dataset:
    #         data['test_result_score'] = result_score
    #     return result_score, dataset

    def item_calculate(self, data, record, output_lang):
        # 個々のoutputの長さをスコアとする
        item_score = len(data['formatted_output'])
        self.item_scores.append(item_score)
        return item_score
    
    def total_calculate(self, dataset, record, output_lang):
        if self.item_scores:
            total_score = sum(self.item_scores) / len(self.item_scores)
        else:
            total_score = 0
        return total_score


# =====================
# Evaluation Evaluator
# =====================

class AccuracyEvaluator(Evaluator):
    # 正確性評価用のEvaluatorクラスの実装は省略


    # def calculate(self, dataset, record):
        
    #     # 正解データと予測データのリストを準備
    #     references = [d['reference'] for d in dataset]
    #     candidates = [d['model_output'] for d in dataset]
    #     # accuracy スコアを計算
    #     score = self.metric.compute(predictions=candidates, references=references)['accuracy']

    #     for data in dataset:
    #         data['accuracy_score'] = score

    #     return score, dataset

    def item_calculate(self, data, record, output_lang):
        return None
    
    def total_calculate(self, dataset, record, output_lang):
        predictions = [int(data['model_output']) for data in dataset]
        references = [int(data['reference']) for data in dataset]
        total_score = self.metric.compute(predictions=predictions, references=references)['accuracy']
        return total_score


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

    # 日本語用のtokenizer
    # Python: 正規表現による簡易版形態素解析
    # https://qiita.com/kinoshita_yuri/items/e15f143981f1616994ed
    def tokenize_ja(text):
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
    
    def item_calculate(self, data, record, output_lang):
        predictions = [data['formatted_output']]
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
        return CodeEvalEvaluator("code_eval", "pass@1", args)
    elif metric_id == "exact_match":
        return ExactMatchEvaluator("exact_match", "exact_match", args)
    else:
        print(f"Unknown metric path: {metric_id}")

class ComposeEvaluator(Evaluator):
    def __init__(self):
        self.evals = []

    def __repr__(self):
        return ','.join(repr(x) for x in self.evals)

    def append(self, eval):
        self.evals.append(eval)
    
    def score(self, records):
        results = {}
        for eval in self.metrics:
            results.update(eval.score(records))
        return results

def compose_evaluator(args):
    metrics = args['metrics']
    if metrics is None:
        return None
    composed = ComposeEvaluator()
    for metric_id in metrics:
        eval = load_evaluator(metric_id, args)
        if eval:
            composed.append(eval)
    return composed