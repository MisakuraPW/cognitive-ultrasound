from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyState:
    posterior_particles: Any
    current_lines: Any
    seed: Any


class Sampler(Protocol):
    def select_action(self, observation, state: PolicyState):
        """Return (next k-hot action, next mask, optional saliency). No GT access."""
        ...


class OfficialSampler:
    """Delegate random, rolling uniform, or CASL to its unmodified upstream callable."""

    def __init__(self, action_selection):
        self.action_selection = action_selection

    def select_action(self, observation, state):
        return self.action_selection(state.posterior_particles, state.current_lines, state.seed)

    def bridge(self, particles, current_lines, seed):
        return self.select_action(None, PolicyState(particles, current_lines, seed))
