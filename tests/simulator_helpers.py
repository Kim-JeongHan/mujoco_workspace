"""Small fakes for simulator and viewer integration tests."""

from contextlib import nullcontext

from mujoco_lab import SimulatorManager


def manager_with(simulator, name="simulator"):
    manager = SimulatorManager()
    manager.add_simulator(name, simulator)
    return manager


class PassiveViewer:
    def __init__(self, steps=1, *, on_lock=None, on_sync=None):
        self.steps = steps
        self.on_lock = on_lock
        self.on_sync = on_sync
        self.sync_calls = 0
        self.closed = False
        self._sim = lambda: None

    def is_running(self):
        return self.sync_calls < self.steps

    def lock(self):
        if self.on_lock is not None:
            self.on_lock()
        return nullcontext()

    def sync(self, *, state_only=False):
        assert state_only
        self.sync_calls += 1
        if self.on_sync is not None:
            self.on_sync()

    def close(self):
        self.closed = True
