from abc import ABC, abstractmethod
# =========================
# Abstract Base Classes
# =========================

class Model(ABC):
    # @abstractmethod
    # def generate(self, *args, **kwargs):
    #     """Abstract method to chat with the model"""
    #     pass
    
    @abstractmethod
    def chat(self, *args, **kwargs):
        """Abstract method to chat with the model"""
        pass