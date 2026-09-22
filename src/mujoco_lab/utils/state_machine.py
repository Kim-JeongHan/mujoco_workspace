"""Small explicit state-transition helper."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping
from typing import Generic, TypeVar

State = TypeVar("State", bound=Hashable)


class StateMachine(Generic[State]):
    """Hold one state and reject transitions outside a fixed graph.

    Transitions are explicit: failed surrounding work does not roll back the state.
    """

    def __init__(self, state: State, transitions: Mapping[State, Collection[State]]) -> None:
        self._state = state
        self._transitions = {
            source: frozenset(destinations) for source, destinations in transitions.items()
        }

    @property
    def state(self) -> State:
        """Return the current state."""
        return self._state

    def transition(self, state: State) -> None:
        """Move to an allowed state or raise RuntimeError."""
        if state not in self._transitions.get(self._state, ()):
            raise RuntimeError(f"Invalid state transition: {self._state!r} -> {state!r}")
        self._state = state
