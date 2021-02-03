from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def response(self, x):
        ...

    @abstractmethod
    def sensitivity(self, x):
        ...
