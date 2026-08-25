"""
Base AI model.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def chat(self, state, message):
        """
        Generate a response using conversation state.
        """
        pass

    @abstractmethod
    def complete(self, system_prompt, message):
        """
        Generate a response without conversation state.
        """
        pass