
import torch

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

class HuggingFaceEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS = {"temperature": 0.7,
                            "do_sample": True,
                            "repetition_penalty": 1.3,
                            "max_new_tokens": 100,
                            "use_cache": True}
    CRITERIA = {"accuracy": 'Score 1: The answer is completely unrelated to the reference. \
Score 3: The answer has minor relevance but does not align with the reference. \
Score 5: The answer has moderate relevance but contains inaccuracies. \
Score 7: The answer aligns with the reference but has minor omissions. \
Score 10: The answer is completely accurate and aligns perfectly with the reference. \
Only respond with a numberical score.'}
    PROMPT = PromptTemplate.from_template('Instruct: Please act as an impartial judge \
and evaluate the quality of the response provided by an AI assistant to the user question displayed below. \
Use this criteria for juding the response {criteria} The true answer is the following: {reference}. \
Begin your evaluation by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale of 1 to 10 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". \
This is the question: {input}. And this is the response: {prediction}. How would you rate this response? \
Remember you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]" \
\nOutput:')

    def __init__(self,
                 model_name: str = None,
                 api_token: str = None,
                 true_answer: str = None,
                 question_asked: str = None,
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 device: str = "cpu"):
        """
        :param model_name: The name of the model.
        :param api_token: The API token for HuggingFace. Default is None.
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if not model_name:
            raise ValueError("model_name must be supplied with init.")
        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.api_token = api_token
        self.true_answer = true_answer
        self.question_asked = question_asked
        self.model_kwargs = model_kwargs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        torch.set_default_device(device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              device_map=self.device,
                                                              trust_remote_code=True,
                                                              pad_token_id=self.tokenizer.pad_token_id,
                                                              eos_token_id=self.tokenizer.eos_token_id)

            model_pipeline = pipeline("text-generation",
                                      model=self.model,
                                      tokenizer=self.tokenizer,
                                      **self.model_kwargs)
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")

        self.model = HuggingFacePipeline(pipeline=model_pipeline)

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.model,
            prompt=self.PROMPT
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])