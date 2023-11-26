import json
import re
import ast

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

    def collate(self, prompt:str, model_output:str) -> str:
        """Collates the model output based on the language and format specified in the template data."""
        output_lang = self.template_data.get('output_lang', '')
        output_format = self.template_data.get('format', 'default')

        if output_format == 'default':
            if output_lang in ['NL', 'en', 'ja', 'ko']:
                formatted_output = self.format_natural_language(prompt, model_output)# 自然言語の整形処理
            elif output_lang in ['PL', 'py', 'cpp', 'js', 'ru']:
                formatted_output = self.format_programming_language(prompt, model_output)# プログラミング言語の整形処理
            else:
                raise ValueError(f"Unsupported output language: {output_lang}")
        
        elif output_format == 'humaneval':
            formatted_output = self.format_humaneval(prompt, model_output)# humanevalの整形処理
        else:
            raise ValueError(f"Unsupported output format: {format}")

        return formatted_output
    
    # collate-subfunction
    ## 自然言語の整形処理
    def format_natural_language(self, prompt, model_output):
        """Formats the natural language text according to specific rules."""
        extracted_output = self.extract_triple_quoted_text(model_output)
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
        """Removes lines that contain the same text as the prompt."""
        prompt_lines = set(prompt.splitlines())
        text_lines = text.splitlines()
        filtered_lines = [line for line in text_lines if line not in prompt_lines]
        return '\n'.join(filtered_lines)

    ## プログラミング言語の整形処理
    def format_programming_language(self, prompt, model_output):
        """Formats the programming language code according to specific rules."""
        extracted_output = self.extract_code_blocks(model_output) + "\n"
        formatted_output = self.extract_functions(extracted_output)
        return formatted_output

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
        """Extracts functions from a code string."""
        pattern_function = r"(def\b.*?)(?=\n\s*def\b|\n\s*$)"
        functions = re.findall(pattern_function, code, re.DOTALL)
        filtered_functions = [func for func in functions if 'return' in func]
        return '\n'.join(filtered_functions).strip()
    
    ## humanevalの整形処理
    def format_humaneval(self, prompt, model_output):
        """Collates the model output for the humaneval format."""
        # if not model_output.startswith("    "):
        #     model_output = "    " + model_output
        combined_output = prompt + "\n" + model_output + "\n"
        return self.extract_functions(combined_output)



# =====================
# Utility Function
# =====================

def load_template(template_path):
    template = TemplateProcessor(template_path)
    return template