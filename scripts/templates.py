from typing import List
import json
import re
import textwrap

class TemplateProcessor:
    def __init__(self, template_path):
        self.template_path = template_path
        self.template_data = self.load()
        self.template_string = self.template_data.get("template", "")
        self.reference_string = self.template_data.get("reference", "")

    def load(self):
        """Loads the template file."""
        if self.template_path.endswith('.json'):
            with open(self.template_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            raise ValueError("Unsupported template format. Please provide a .json file.")

    def process(self, data):
        """Creates a prompt using the loaded template and provided data."""
        try:
            prompt = self.template_string.format(**data)
            return prompt
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for template: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in template formatting: {e}")
    
    def process_reference(self, data):
        """Creates a reference using the loaded template and provided data."""
        try:
            reference = self.reference_string.format(**data)
            return reference
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for reference: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in reference formatting: {e}")
    
    def collate(self, prompt:str, model_output_list:list) -> list:
        """Collates the model output based on the language and format specified in the template data."""
        output_lang = self.template_data.get('output_lang', '')
        output_format = self.template_data.get('format', 'default')
        begin_marker = self.template_data.get('begin_marker', None)
        end_marker = self.template_data.get('end_marker', None)
        
        formatted_output_list = []
        format_checked_list = []

        for model_output in model_output_list:
            
            if begin_marker and end_marker:
                start_index = model_output.find(begin_marker)
                end_index = model_output.find(end_marker)
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    model_output = model_output[start_index + len(begin_marker):end_index]
                    format_checked_list.append(1)
                else:
                    format_checked_list.append(0)

            if output_format == 'default':
                if output_lang in ['NL', 'en', 'ja', 'ko']:
                    formatted_output = self.format_natural_language(prompt, model_output)# 自然言語の整形処理
                elif output_lang in ['PL', 'py', 'cpp', 'js', 'ru']:
                    formatted_output = self.format_programming_language(prompt, model_output)# プログラミング言語の整形処理
                else:
                    raise ValueError(f"Unsupported output language: {output_lang}")
            
            elif 'xml_' in output_format:
                formatted_output = self.format_xml(output_format, model_output)# xmlタグ形式の整形処理

                if formatted_output['output'] is not None:
                    if output_lang in ['NL', 'en', 'ja', 'ko']:  # NL: promptに含まれる文字列のremove
                        formatted_output['output'] = self.remove_prompt_lines(prompt, formatted_output['output'])
                    elif output_lang in ['PL', 'py', 'cpp', 'js', 'ru']:  # PL: 余分なインデント削除
                        formatted_output['output'] = self.remove_leading_whitespace(formatted_output['output'])
                    else:
                        raise ValueError(f"Unsupported output language: {output_lang}")
            
            elif output_format == 'humaneval':
                formatted_output = self.format_humaneval(prompt, model_output)# humanevalの整形処理

            elif output_format == 'multiplechoice':
                formatted_output = self.format_multiplechoice(prompt, model_output)
            
            elif output_format == 'tag':
                formatted_output = self.extract_tag(model_output)

            else:
                raise ValueError(f"Unsupported output format: {format}")
            
            formatted_output_list.append(formatted_output)

        return output_lang, output_format, formatted_output_list, format_checked_list
    

    # collate-subfunction
    ## 自然言語の整形処理
    def format_natural_language(self, prompt, model_output):
        """Formats the natural language text according to specific rules."""
        extracted_output = self.extract_triple_quoted_text(model_output) #TODO：プロンプト指示に合わせて抽出する
        formatted_output = self.remove_prompt_lines(prompt, extracted_output)
        return formatted_output

    def extract_triple_quoted_text(self, text):
        """Extracts text enclosed in triple quotes."""
        pattern_triplequotes = r'""".*?"""'
        matches = re.findall(pattern_triplequotes, text, re.DOTALL)
        if matches:
            extracted_content = [match[3:-3] for match in matches]
            extracted_text = '\n'.join(extracted_content)
        else:
            extracted_text = text
        return extracted_text
    
    def remove_prompt_lines(self, prompt, text):
        """Removes lines that contain the same text as the prompt, ignoring leading whitespace."""
        prompt_lines = set(line.strip() for line in prompt.splitlines())
        text_lines = text.splitlines()
        filtered_lines = [line for line in text_lines if line.strip() not in prompt_lines]
        return '\n'.join(filtered_lines)


    ## プログラミング言語の整形処理
    def format_programming_language(self, prompt, model_output):
        """Formats the programming language code according to specific rules."""
        extracted_output = self.extract_code_blocks(model_output) + "\n"
        # extracted_output = self.extract_first_code_block(model_output) + "\n"
        formatted_output = self.extract_functions(extracted_output)
        formatted_output = self.remove_leading_whitespace(formatted_output)
        return formatted_output

    # def extract_first_code_block(self, text):
    #     """Extracts text enclosed in the first code block."""
    #     pattern_codeblock = r'```.*?```'
    #     match = re.search(pattern_codeblock, text, re.DOTALL)
    #     if match:
    #         extracted_text = match.group()[3:-3]
    #     else:
    #         extracted_text = text
    #     return extracted_text

    def extract_code_blocks(self, text):
        """Extracts text enclosed in code blocks."""
        pattern_codeblock = r'```.*?```'
        matches = re.findall(pattern_codeblock, text, re.DOTALL)
        if matches:
            extracted_content = [match[3:-3] for match in matches]
            extracted_text = '\n'.join(extracted_content)
        else:
            extracted_text = text
        return extracted_text

    def extract_functions(self, code:str):
        # """Extracts functions from a code string."""
        # pattern_function = r"(def\b.*?)(?=\n\s*def\b|\n\s*$)"
        # functions = re.findall(pattern_function, code, re.DOTALL)
        # filtered_functions = [func for func in functions if 'return' in func]
        # return '\n'.join(filtered_functions).strip()
        codelines = code.split('\n')
        import_sentenses = [c for c in codelines if c.startswith('from') or c.startswith('import')]

        pattern_function = r"(def\b.*?)(?=\n\s*def\b|\n\s*$)"
        functions = re.findall(pattern_function, code, re.DOTALL)
        filtered_functions = [func for func in functions if 'return' in func]
        return '\n'.join(import_sentenses).strip() + '\n\n' +'\n'.join(filtered_functions).strip()
    
    def remove_leading_whitespace(self, code:str):
        """Remove leading whitespace (common minimum indent) from each line in the given code."""
        return textwrap.dedent(code).strip()


    # xml形式の整形処理
    def format_xml(self, output_format, model_output):
        """Collates the model output for the xml format."""
        tag_name = output_format.split("_")[1] # output_format:xml_code → </code>
        stop_sequence = f"</{tag_name}>"
        stop_index = model_output.find(stop_sequence)

        if stop_index != -1:  # stop_sequence が見つかった場合
            formatted_output = model_output[:stop_index]
            return {"formatted_correctly": 1, "output": formatted_output}
        else:
            return {"formatted_correctly": 0, "output": None}
        
    # tag形式の整形処理
    def extract_tag(self, model_output):
        pattern_tag = r"\[PYTHON\](.*?)\[/PYTHON\]"
        match = re.search(pattern_tag, model_output, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
        else:
            extracted_text = model_output
        return extracted_text

class TemplateProcessor:
    def __init__(self, args):
        self.prompt = args['prompt_template|=']
        self.reference = args['reference_template|=']
        self.begin = args['extract_begin']
        self.end = args['extract_end']

    def create_prompt(self, data):
        """Creates a prompt using the loaded template and provided data."""
        try:
            prompt = self.prompt.format(**data)
            return prompt
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for template: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in template formatting: {e}")
    
    def create_reference(self, data):
        """Creates a reference using the loaded template and provided data."""
        try:
            reference = self.reference.format(**data)
            return reference
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for reference: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in reference formatting: {e}")

    def extract(self, text:str) -> str:
        if isinstance(text, list):
            return [self.extract(t) for t in text]
        if self.begin or self.end:
            lines = text.splitlines()
            extracted = []
            inclusion = False if self.begin else True
            for line in lines:
                if self.end and line.startswith(self.end):
                    break
                if self.begin and line.startswith(self.begin):
                    inclusion = True
                if inclusion:
                    extracted.append(line)
            return '\n'.join(extracted)
        return text

def has_all_keys(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def guess_template(data: dict, args):
    if has_all_keys(data, 'prompt|test|entry_point'):
        return {
            "prompt_template": "{prompt}",
            "reference_template": "\n{test}\ncheck({entry_point})\n"
        }
    output_delim = '###Output\n'
    if has_all_keys(data, 'question|choice0|choice1|choice2|choice3|choice4|label'):
        return {
            "prompt_template": "{question}\n選択肢: (0) {choice0} (1) {choice1} (2) {choice2} (3) {choice3} (4) {choice4}\n" + output_delim,
            "reference_template": "{label}",
            "output_format": "line",
        }
    return {}

# =====================
# Utility Function
# =====================

def load_template(args: dict, dataset:List[dict]):
    template_path = args['template|template_path']
    if template_path:
        args.load_config(template_path, overwrite=False)
    if 'prompt_template' not in args:
        # テンプレート推論を試みる
        config = guess_template(dataset[0], args)
        args.update(config, overwrite=False)
    template = TemplateProcessor(args)
    args.verbose_print(f'プロンプトを確認してね//Confirm the prompt format\n{template.create_prompt(dataset[0])}')
    args.verbose_print(f'参照データも確認してね//Confirm the reference data\n{template.create_reference(dataset[0])}')
    return template
    