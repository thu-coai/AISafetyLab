from abc import ABC, abstractmethod
# =========================
# Abstract Base Classes
# =========================

class Defender(ABC):
    @abstractmethod
    def defend(self, *args, **kwargs):
        """Abstract method to perform defense."""
        pass

class PreprocessDefender(Defender):
    @abstractmethod
    def defend(self, messages):
        """Defend method for preprocessing defenders.

        Args:
            messages (str / list): The original input messages.

        Returns:
            list: The defended (processed) input messages.
            bool: whether to reject
        """
        pass

class IntraprocessDefender(Defender):
    @abstractmethod
    def defend(self, model, messages):
        """Defend method for intraprocess defenders.

        Args:
            messages (str / list): The input text.
            model (callable): The language model used for generation.

        Returns:
            str: The generated text after defense.
        """
        pass

class PostprocessDefender(Defender):
    @abstractmethod
    def defend(self, generated_text, query):
        """Defend method for postprocessing defenders.

        Args:
            generated_text (str): The text generated by the model.

        Returns:
            str: The defended (filtered) output text.
        """
        pass