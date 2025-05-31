import abc
import numpy as np

class EmbedderModel(abc.ABC):
    """Abstract base class for embedders."""
    
    @abc.abstractmethod
    def embed(self, language: str, texts: list[str]) -> np.ndarray:
        pass
