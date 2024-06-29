from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def PotentialFactory(type: str) -> PotentialFunction:
    if type == "kronig-penney":
        return KronigPenneyPotential()
    elif type == "sine":
        return SinePotential()
    else:
        raise ValueError(f"Unknown potential form: {type}")


class PotentialFunction(ABC):
    amplitude = 1.0
    width = 0.5
    form = ""

    @abstractmethod
    def apply(self, coordinates: list[float]) -> float:
        """docstring"""

    def __str__(self) -> str:
        return self.form

    def __repr__(self) -> str:
        return self.form

    def __call__(self, coordinates: list[float]) -> float:
        return np.round(self.apply(coordinates), 6)


class KronigPenneyPotential(PotentialFunction):
    form = "V(x) = V_0 if x <= 0.5 else -V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        return self.amplitude if x <= 0.5 else -self.amplitude


class SinePotential(PotentialFunction):
    form = "V(x) = V_0 sin(2 pi x)"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        return self.amplitude * np.sin(2 * np.pi * x)
