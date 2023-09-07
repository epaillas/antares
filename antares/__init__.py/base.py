from abc import ABC, abstractmethod


class BaseSummary(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.__repr__()