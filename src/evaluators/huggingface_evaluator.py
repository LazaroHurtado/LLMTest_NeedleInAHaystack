
import os

from .evaluator import Evaluator

from huggingface_hub import login
from langchain.evaluation import load_evaluator
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

class HuggingFaceEvaluator(Evaluator):
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = None,
                 api_token: str = None,
                 true_answer: str = None,
                 question_asked: str = None,
                 model_kwargs: dict = {"temperature": 0.0}):
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
        if (not api_token) and (not os.getenv('HUGGINGFACEHUB_API_TOKEN')):
            raise ValueError("HUGGINGFACEHUB_API_TOKEN must be in env. Used for evaluation model")

        self.model_name = model_name
        self.api_token = api_token or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.true_answer = true_answer
        self.question_asked = question_asked
        self.model_kwargs = model_kwargs

        login(token=self.api_token)

        self.model = HuggingFaceHub(repo_id=self.model_name,
                                    task="text-generation",
                                    model_kwargs=self.model_kwargs)
        self.evaluator = ChatHuggingFace(llm=self.model)

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
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