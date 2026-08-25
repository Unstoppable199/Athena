"""
Base capability service.
"""

from abc import ABC, abstractmethod


class BaseService(ABC):

    @abstractmethod
    def execute(self, request: str):
        """
        Execute a user request using this capability.

        Returns:

        {
            "success": bool,
            "data": ...,
            "error": ...
        }
        """
        pass