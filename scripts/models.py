from typing import List
import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from adhoc import AdhocArguments

try:
    from openai import OpenAI
except ModuleNotFoundError:
    ## モジュールが見つからない場合は、
    ## OpenAIを実行するまでエラーを出さない
    OpenAI = None

try:
    import boto3
except ModuleNotFoundError:
    ## モジュールが見つからない場合は、
    ## boto3を実行するまでエラーを出さない
    boto3 = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================
# Base Classes
# =====================

class Model(object):
    def __init__(self, model_path, args):
        """
        Base class for abstracting a pretrained model.
        """
        self.model_path = model_path
        self.args = args
        self.num_sequences = self.args['num_return_sequences|n|N|=1']

    def __repr__(self):
        return self.model_path
    
    def generate_list(self, prompt: str, n=1) -> List[str]:
        return [self.generate_text(prompt) for _ in range(n)]

class TestModel(Model):

    def generate_list(self, prompt: str, n=1) -> List[str]:
        test_results = [f"{prompt}\n###Output\n{i}\n" for i in range(n)]
        return test_results

class OpenAIModel(Model):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        if OpenAI is None:
            args.raise_uninstalled_module('openai')
        # Default arguments for OpenAI API
        default_args = {
            "temperature": args['temperature|=0.2'],
            "top_p": args['top_p|=0.95'],
            "max_tokens": args['max_tokens|max_length|=512'], 
        }
        self.openai_api_key = args['openai_api_key|api_key|!error']
        self.model_args = default_args

    def generate_list(self, prompt: str, n=1) -> List[str]:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            **self.model_args
        )
        responses = [choice.message.content for choice in response.choices]
        return responses

class BedrockModel(Model):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        if boto3 is None:
            args.raise_uninstalled_module('boto3')

        default_args = {
            "max_tokens_to_sample": args['max_tokens|max_length|=512'],
            "temperature": args['temperature|=0.2'],
            "top_p": args['top_p|=0.95'],
        }
        self.aws_access_key_id = args['aws_access_key_id']
        self.aws_secret_access_key = args['aws_secret_access_key']
        self.model_args = default_args
    
    def check_and_append_claude_format(self, prompt: str) -> str:
        ## FIXME: 改行の位置はここでいいのか？
        human_str = "\n\nHuman:"
        assistant_str = "\n\nAssistant:"

        if human_str not in prompt:
            prompt = human_str + prompt

        if assistant_str not in prompt:
            prompt += assistant_str

        return prompt

    def generate_text(self, prompt: str) -> str:
        bedrock = boto3.client("bedrock-runtime",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name='ap-northeast-1'
        )
        prompt = self.check_and_append_claude_format(prompt)
        body = json.dumps(
            {
                "prompt": prompt,
                "anthropic_version": "bedrock-2023-05-31",
                **self.model_args,
            }
        )
        response = bedrock.invoke_model(body=body, modelId=self.model_path)
        response_body = json.loads(response.get("body").read())
        return response_body.get("completion")


class HFModel(Model):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_auth_token=args['hf_token'],
            trust_remote_code=True, 
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if args['use_4bit']:
            model = load_4bit_model(model_path, args)
        else:
            model = load_normal_model(model_path, args)
        
        if "max_new_tokens" in args:
            self.generator_args = {
                "max_new_tokens": args['max_new_tokens'],
                "do_sample": True,
                "temperature": args['temperature|=0.2'],
                "top_p": args['top_p|=0.95'],
                "return_full_text": False,
            }
        else:
            self.generator_args = {
                "max_length": args['max_length|=512'],
                "do_sample": True,
                "temperature": args['temperature|=0.2'],
                "top_p": args['top_p|=0.95'],
                "return_full_text": False,
#                "num_return_sequences": self.num_sequences,
            }

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            # device=0 if torch.cuda.is_available() else -1,
            use_auth_token=args['hf_token'],
            # 何が指定できるのか？？？
            # **generator_args
        )
    
    def generate_list(self, prompt: str, n=1) -> List[str]:
        # pipelineなしで実装----------------------------------
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # generated_ids = self.model.generate(input_ids, **self.model_args)
        # return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # ----------------------------------
        generated_texts = self.generator(prompt, 
                                         ### ここは何を指定するのか？
                                        num_return_sequences = n,
                                         **self.generator_args, 
                                         pad_token_id=self.generator.tokenizer.eos_token_id)
        generated_texts_list = [item['generated_text'] for item in generated_texts]
        return generated_texts_list


def load_normal_model(model_path, args):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_auth_token=args['hf_token'],
            trust_remote_code=True,
            device_map="auto",
        )
        return model
    except BaseException as e:
        print(f'Unable to load HuggingFace Model: {model_path}')
        raise e
        sys.exit(1)

def load_4bit_model(model_path, args):
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            quantization_config=bnb_config,
            device_map="auto", 
            trust_remote_code=True,
            use_auth_token=args['hf_token'],
        )
        return model
    except BaseException as e:
        print(f'4ビット量子化モデルがロードできまえん//Unable to load 4Bit Quantimization Model: {e}')
        print('とりあえず、ノーマルモデルを試します//Trying normal model...')
        return load_normal_model(model_path, args)
        
def load_model(args):
    model_path = args['model_path']
    try:
        if model_path is None:
            return TestModel('dummy/model', args)
        elif model_path.startswith("openai:"):
            return OpenAIModel(model_path[7:], args)
        elif model_path.startswith("bedrock:"):
            return BedrockModel(model_path[8:], args)
        else:
            return HFModel(model_path, args)
    except Exception as e:
        print(f"Failed to load the model. Error message: {e}")
        raise e
