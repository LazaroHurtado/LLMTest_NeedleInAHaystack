
from abc import ABC, abstractmethod

class Evaluator(ABC):
    CRITERIA: str | dict[str, str]

    @abstractmethod
    def evaluate_response(self, response: str) -> int: ...