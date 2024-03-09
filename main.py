
from src import LLMNeedleHaystackTester, OpenAIEvaluator
from src import ModelTester, AnthropicTester, HuggingFaceTester, OpenAITester
from src import Evaluator, HuggingFaceEvaluator, OpenAIEvaluator

from dataclasses import dataclass
from dotenv import load_dotenv
from jsonargparse import CLI
from typing import Optional

load_dotenv()

@dataclass
class CommandArgs():
    provider: str = "openai"
    evaluator: str = "openai"
    model_name: str = "gpt-4-1106-preview"
    evaluator_name: str = "gpt-4"
    api_key: Optional[str] = None
    evaluator_api_key: Optional[str] = None

def get_model_to_test(args) -> ModelTester:
    match args.provider.lower():
        case "openai":
            return OpenAITester(model_name=args.model_name, api_key=args.api_key)
        case "anthropic":
            return AnthropicTester(model_name=args.model_name, api_key=args.api_key)
        case "huggingface" | "hf":
            return HuggingFaceTester(model_name=args.model_name, device="cpu")
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")
        
def get_evaluator(args, question: str, answer: str) -> Evaluator:
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(model_name=args.evaluator_name,
                                   question_asked=question,
                                   true_answer=answer,
                                   api_key=args.evaluator_api_key)
        case "huggingface" | "hf":
            return HuggingFaceEvaluator(model_name=args.evaluator_name,
                                        api_token=args.evaluator_api_key,
                                        question_asked=question,
                                        true_answer=answer)
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    args = CLI(CommandArgs, as_positional=False)

    needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    retrieval_question = "What is the best thing to do in San Francisco?"

    model_to_test = get_model_to_test(args)
    evaluator = get_evaluator(args, retrieval_question, needle)

    tester = LLMNeedleHaystackTester(model_to_test=model_to_test,
                                     evaluator=evaluator,
                                     needle=needle,
                                     retrieval_question=retrieval_question,
                                     context_lengths_min=500,
                                     context_lengths_max=2048,
                                     results_version=2)
    tester.start_test()

if __name__ == "__main__":
    main()