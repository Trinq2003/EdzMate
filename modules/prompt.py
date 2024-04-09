from transformers import AutoTokenizer

class Prompt:
    def __init__(self, prompt_template, tokenizer: AutoTokenizer) -> None:
        self.prompt_template = tokenizer.apply_chat_template(prompt_template)