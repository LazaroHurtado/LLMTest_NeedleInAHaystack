import torch

from .model_tester import ModelTester

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from typing import Optional

class HuggingFaceTester(ModelTester):
    DEFAULT_MODEL_KWARGS = {"temperature": 0.0,
                            "do_sample": True,
                            "repetition_penalty": 1.3,
                            "max_new_tokens": 500,
                            "use_cache": True,}

    def __init__(self,
                 model_name: str,
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 device: str = None):
        self.model_name = model_name

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.enc = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")
        
        self.model_kwargs = model_kwargs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)

        with open('HuggingFace_prompt.txt', 'r') as file:
            self.prompt_structure = file.read()

    async def evaluate_model(self, prompt: str) -> str:
        tokens = self.enc(prompt, return_tensors="pt")["input_ids"]
        response = await self.model.generate(tokens, **self.model_kwargs)
        return response.data.tolist()[0]
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        return self.prompt_structure.format(
            retrieval_question=retrieval_question,
            context=context) 
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        return self.enc(text, truncation=True)["input_ids"]
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        return self.enc.decode(tokens[:context_length])