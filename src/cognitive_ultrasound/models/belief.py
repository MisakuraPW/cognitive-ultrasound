from typing import Any, Protocol


class BeliefModel(Protocol):
    def update(self, observation: Any, mask: Any, state: Any) -> Any:
        """Update a posterior state from measured information only."""
        ...
