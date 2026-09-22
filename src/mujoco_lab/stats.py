"""Per-run tracking and actuator saturation statistics collected by Simulator."""


class RunStats:
    def __init__(self) -> None:
        self.steps = 0
        self.saturated_steps = 0
        self.errors: list[float] = []

    def update(self, saturated: bool, error: float | None) -> None:
        self.steps += 1
        self.saturated_steps += int(saturated)
        if error is not None:
            self.errors.append(error)

    def describe(self, error_label: str, unit: str) -> str:
        lines = [f"steps: {self.steps}"]
        if self.steps:
            share = 100.0 * self.saturated_steps / self.steps
            lines.append(f"torque saturation: {share:.1f}% of steps")
        if self.errors:
            mean = sum(self.errors) / len(self.errors)
            maximum = max(self.errors)
            lines.append(f"{error_label}: mean {mean:.4f} {unit}, max {maximum:.4f} {unit}")
        return "\n".join(lines)
