from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, TypeVar

T = TypeVar('T')


@dataclass(frozen=True, init=True, repr=True, eq=False, unsafe_hash=None, kw_only=True)
class Serializable:
    """Interface for a serializable object."""

    _cache: Dict[str, Any] = field(init=False, repr=False, compare=False, hash=None, kw_only=True, default_factory=dict)
    """Internal structure to handle lazy initializations and other cached values."""

    def _lazy_initialization(self, attribute: str, function: Callable[[], T]) -> T:
        """Checks if a value is in the cache, otherwise initializes and stores it."""
        value = self._cache.get(attribute, None)
        if value is None:
            value = function()
            self._cache[attribute] = value
        return value

    @property
    @abstractmethod
    def configuration(self) -> Dict[str, Any]:
        """Returns a JSON-compliant configuration which uniquely identifies the object."""
        pass

    @property
    def key(self) -> str:
        """Returns the full name of an object based on its configuration."""
        return '-'.join([str(v) for v in self.configuration.values()])

    def __eq__(self, other: 'Serializable') -> bool:
        return self.configuration == other.configuration

    def __hash__(self) -> int:
        return hash(self.key)
