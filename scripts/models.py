import os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

try:
    from openai import OpenAI
except ModuleNotFoundError:
    ## モジュールが見つからない場合は、
    ## 実行するまでエラーを出さない
    OpenAI = None

try:
    import boto3
except ModuleNotFoundError:
    ## モジュールが見つからない場合は、
    ## 実行するまでエラーを出さない
    boto3 = None



os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================
# Base Classes
# =====================

class Model:
    """Base class for abstracting a pretrained model."""
    def generate(self, prompt: str)->str:
        return f"Generated response for: {prompt}"


class ModelLoader:
    """Loads a Model instance based on a model name and additional arguments."""
    def __init__(self, model_name, model_args:dict):
        self.model_name = model_name
        self.model_args = model_args

    def load(self)->Model:
        return Model()


# =====================
# Testing Code
# =====================

class TestModel(Model):
    def __init__(self, model_name, model_args=None):
        default_args = {"n": 1}
        model_args = model_args or {}
        default_args.update(model_args)

        super().__init__()
        self.model_args = default_args

    def generate(self, prompt: str) -> list:
        num_sequences = self.model_args.get('num_return_sequences') or self.model_args.get('n') or 1
        test_results = [
            f"Response #{i}: Generated response for: {prompt} \n with args: {self.model_args}"
            for i in range(num_sequences)
        ]
        return test_results

class TestModelLoader(ModelLoader):
    def __init__(self, model_name, model_args=None):
        super().__init__(model_name, model_args)

    def load(self) -> TestModel:
        return TestModel(self.model_name, self.model_args)


# =====================
# HuggingFace Model Integration
# =====================

class HFModel(Model):
    def __init__(self, model_name, hf_token=None, model_args=None, quantize=False):
        default_args = {
            "max_length": 512,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.2,
            "return_full_text": False,
            "num_return_sequences": 1,
        }
        model_args = model_args or {}
        if "max_new_tokens" in model_args:
            default_args.pop("max_length", None)
        default_args.update(model_args)

        # super().__init__()
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token if hf_token else None,
            trust_remote_code=True, 
            padding_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # pipelineなしで実装----------------------------------
        # # Initialize the model
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, 
        #     use_auth_token=hf_token if hf_token else None,
        #     trust_remote_code=True
        # )

        # # Set the device to GPU if available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        # ----------------------------------

        self.model_args = default_args

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=bnb_config,
                device_map="auto", 
                trust_remote_code=True,
                use_auth_token=hf_token if hf_token else None,
            )
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_name, device_map="auto", use_auth_token=hf_token if hf_token else None, load_in_4bit=True
            # )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                use_auth_token=hf_token if hf_token else None, 
                trust_remote_code=True,
                device_map="auto",
            )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=0 if torch.cuda.is_available() else -1,
            use_auth_token=hf_token if hf_token else None,
            **self.model_args
        )

    
    def generate(self, prompt: str) -> list:
        # pipelineなしで実装----------------------------------
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # generated_ids = self.model.generate(input_ids, **self.model_args)
        # return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # ----------------------------------
        generated_texts = self.generator(prompt, **self.model_args, pad_token_id=self.generator.tokenizer.eos_token_id)
        generated_texts_list = [item['generated_text'] for item in generated_texts]
        return generated_texts_list

class HFModelLoader(ModelLoader):
    def __init__(self, model_name, hf_token=None, model_args=None, quantize=True):
        super().__init__(model_name, model_args)
        self.hf_token = hf_token
        self.quantize = quantize

    def load(self) -> HFModel:
        return HFModel(self.model_name, self.hf_token, self.model_args, self.quantize)


# =====================
# OpenAI Model Integration
# =====================

class OpenAIModel(Model):
    def __init__(self, openai_api_key, model_name, model_args=None):
        # Default arguments for OpenAI API
        default_args = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 512, 
            "n": 1}
        # Override defaults with any user-provided arguments
        model_args = model_args or {}
        default_args.update(model_args)

        super().__init__()
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.model_args = default_args

    def generate(self, prompt: str) -> list:
        client = OpenAI(api_key=self.openai_api_key)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.model_args
        )
        # prompt_and_response = prompt + "\n" + response.choices[0].message.content
        # return response.choices[0].message.content
        responses = [choice.message.content for choice in response.choices]
        return responses

class OpenAIModelLoader(ModelLoader):
    def __init__(self, openai_api_key, model_name, model_args=None):
        super().__init__(model_name, model_args)
        self.openai_api_key = openai_api_key

    def load(self) -> OpenAIModel:

        return OpenAIModel(self.openai_api_key, self.model_name, self.model_args)


# =====================
# Anthropic Model Integration
# =====================

class AnthropicModel(Model):
    def __init__(self, aws_access_key_id, aws_secret_access_key, model_name, model_args=None):
        # Default arguments for Anthropic Claude API
        default_args = {
            "max_tokens_to_sample": 512,
            "temperature": 0.2,
            "top_p": 0.95,
        }
        # Override defaults with any user-provided arguments
        model_args = model_args or {}
        default_args.update(model_args)

        super().__init__()
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.model_name = model_name
        self.model_args = default_args
    
    def check_and_append_claude_format(self, prompt: str) -> str:
        human_str = "\n\nHuman:"
        assistant_str = "\n\nAssistant:"

        if human_str not in prompt:
            prompt = human_str + prompt

        if assistant_str not in prompt:
            prompt += assistant_str

        return prompt

    def generate(self, prompt: str) -> str:
        bedrock = boto3.client("bedrock-runtime",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name='ap-northeast-1'
        )

        prompt = self.check_and_append_claude_format(prompt)

        body = json.dumps(
            {
                "prompt": prompt,
                # "prompt": "\n\nHuman: Tell me a funny joke about outer space\n\nAssistant:",
                "anthropic_version": "bedrock-2023-05-31",
                **self.model_args,
            }
        )

        response = bedrock.invoke_model(body=body, modelId=self.model_name)
        response_body = json.loads(response.get("body").read())
        return response_body.get("completion")

class AnthropicModelLoader(ModelLoader):
    def __init__(self, aws_access_key_id, aws_secret_access_key, model_name, model_args=None):
        super().__init__(model_name, model_args)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

    def load(self) -> AnthropicModel:

        return AnthropicModel(self.aws_access_key_id, self.aws_secret_access_key, self.model_name, self.model_args)


# =====================
# Model Loader Factory
# =====================

class ModelLoaderFactory:
    @staticmethod
    def create(model_name, openai_api_key=None, aws_access_key_id=None, aws_secret_access_key=None, hf_token=None, model_args=None, quantize=True):
        try:
            if model_name == "test":
                return TestModelLoader(model_name, model_args)
            elif model_name.startswith("gpt"):
                return OpenAIModelLoader(openai_api_key, model_name, model_args)
            elif model_name.startswith("anthropic"):
                return AnthropicModelLoader(aws_access_key_id, aws_secret_access_key, model_name, model_args)
            else:
                return HFModelLoader(model_name, hf_token, model_args, quantize)
        except Exception as e:
            print(f"Failed to load the model. Error message: {e}")
            raise e



# =====================
# Utility Function
# =====================

def load_model(model_path, openai_api_key, aws_access_key_id, aws_secret_access_key, hf_token, model_args, quantize):
    model_loader = ModelLoaderFactory.create(
        model_path, 
        openai_api_key, 
        aws_access_key_id, 
        aws_secret_access_key, 
        hf_token, 
        model_args, 
        quantize
    )
    model = model_loader.load()
    return model

######################

class Model:
    def __init__(self, model_path, args):
        """Base class for abstracting a pretrained model."""
        self.model_path = model_path
        self.args = args

    def __repr__(self):
        return self.model_path
    
    def generate(self, prompt: str)->str:
        return f"Generated response for: {prompt}"

class TestModel(Model):

    def generate(self, prompt: str) -> list:
        num_sequences = self.args['num_return_sequences|n|N|=1']
        test_results = [
            f"{prompt}\nResponse #{i}"
            for i in range(num_sequences)
        ]
        return test_results

class OpenAIModel(Model):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        if OpenAI is None:
            print('===============================')
            print('OpenAI module is not installed.')
            print('Try `pip3 install -U openai`')
            sys.exit(1)
        # Default arguments for OpenAI API
        default_args = {
            "temperature": args['temperature|=0.2'],
            "top_p": args['top_p|=0.95'],
            "max_tokens": args['max_tokens|max_length|=512'], 
            "n": args['n|=1'],
        }
        self.openai_api_key = args['openai_api_key|api_key|!error']
        self.model_args = default_args

    def generate(self, prompt: str) -> list:
        client = OpenAI(api_key=self.openai_api_key)
        
        response = client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": prompt}],
            **self.model_args
        )
        responses = [choice.message.content for choice in response.choices]
        return responses

class BedrockModel(Model):
    def __init__(self, model_path, args):
        super().__init__(model_path, args)
        if boto3 is None:
            print('===============================')
            print('Boto3 module is not installed.')
            print('Try `pip3 install -U boto3`')
            sys.exit(1)

        default_args = {
            "max_tokens_to_sample": args['max_tokens|max_length|=512'],
            "temperature": args['temperature|=0.2'],
            "top_p": args['top_p|=0.95'],
        }
        self.aws_access_key_id = args['aws_access_key_id']
        self.aws_secret_access_key = args['aws_secret_access_key']
        self.model_args = default_args
    
    def check_and_append_claude_format(self, prompt: str) -> str:
        human_str = "\n\nHuman:"
        assistant_str = "\n\nAssistant:"

        if human_str not in prompt:
            prompt = human_str + prompt

        if assistant_str not in prompt:
            prompt += assistant_str

        return prompt

    def generate(self, prompt: str) -> str:
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
        
        tokenizer = AutoTokenizer.from_pretrained(
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
                "num_return_sequences": 1,
            }
        else:
            self.generator_args = {
                "max_length": args['max_length|=512'],
                "do_sample": True,
                "temperature": args['temperature|=0.2'],
                "top_p": args['top_p|=0.95'],
                "return_full_text": False,
                "num_return_sequences": 1,
            }

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # device=0 if torch.cuda.is_available() else -1,
            use_auth_token=args['hf_token'],
            # 何が指定できるのか？？？
            # **generator_args
        )
    
    def generate(self, prompt: str) -> list:
        # pipelineなしで実装----------------------------------
        # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # generated_ids = self.model.generate(input_ids, **self.model_args)
        # return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # ----------------------------------
        generated_texts = self.generator(prompt, 
                                         ### ここは何を指定するのか？
                                         **self.generator_args, 
                                         pad_token_id=self.generator.tokenizer.eos_token_id)
        generated_texts_list = [item['generated_text'] for item in generated_texts]
        return generated_texts_list


def load_normal_model(model_path, model_args):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_auth_token=model_args['hf_token'],
            trust_remote_code=True,
            device_map="auto",
        )
        return model
    except BaseException as e:
        print(f'Unable to load HuggingFace Model: {model_path}')
        sys.exit(1)

def load_4bit_model(model_path, model_args):
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
            use_auth_token=model_args['hf_token'],
        )
        return model
    except BaseException as e:
        print(f'Unable to load 4Bit Quantimization Model {e}')
        print('Trying normal model...')
        return load_normal_model(model_path, model_args)
        

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
