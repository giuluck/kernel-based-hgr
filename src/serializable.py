from abc import abstractmethod
from typing import Dict, Any


class Serializable:
    """Interface for a serializable object."""

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Returns the object configuration."""
        pass

    @property
    def fullname(self) -> str:
        """Returns the full name of an object based on its configuration."""
        return '-'.join([str(v) for v in self.config.values()])
