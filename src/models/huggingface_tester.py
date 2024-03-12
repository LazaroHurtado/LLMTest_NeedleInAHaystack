import torch

from .model_tester import ModelTester

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from typing import Optional

class HuggingFaceTester(ModelTester):
    DEFAULT_MODEL_KWARGS = {"temperature": 0.7,
                            "do_sample": True,
                            "repetition_penalty": 1.3,
                            "max_new_tokens": 100,
                            "use_cache": True,}

    def __init__(self,
                 model_name: str,
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs

        torch.set_default_device(device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              device_map="auto",
                                                              trust_remote_code=True,
                                                              pad_token_id=self.tokenizer.pad_token_id,
                                                              eos_token_id=self.tokenizer.eos_token_id)
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")
        

        with open('HuggingFace_finetuned_prompt.txt', 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        prompt = prompt.replace("\n", " ").replace("Output:", "\nOutput:")
        tokens = self.tokenizer(prompt,
                                return_tensors="pt",
                                return_attention_mask=False)["input_ids"]
        
        response = self.model.generate(tokens,
                                       pad_token_id=self.tokenizer.eos_token_id,
                                       **self.model_kwargs)
        encoded_response = response.data.tolist()[0][len(tokens[0]):]

        return self.decode_tokens(encoded_response)
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return self.prompt_structure.format(context=context,
                                            retrieval_question=retrieval_question) 
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, truncation=True)["input_ids"]
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        return self.tokenizer.decode(tokens[:context_length])