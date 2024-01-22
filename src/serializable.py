from abc import abstractmethod
from typing import Dict, Any


class Serializable:
    """Interface for a serializable object."""

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Returns the object configuration."""
        pass
