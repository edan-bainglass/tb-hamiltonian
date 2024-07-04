from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def PotentialFactory(type: str) -> PotentialFunction:
    if type == "null":
        return NullPotential()
    elif type == "kronig-penney":
        return KronigPenneyPotential()
    elif type == "sine":
        return SinePotential()
    else:
        raise ValueError(f"Unknown potential form: {type}")


class PotentialFunction(ABC):
    name = ""
    form = ""
    amplitude = 1.0
    width = 0.5

    @abstractmethod
    def apply(self, coordinates: np.ndarray) -> float:
        """docstring"""

    def __str__(self) -> str:
        return self.form

    def __repr__(self) -> str:
        return self.form

    def __call__(self, coordinates: np.ndarray) -> float:
        return np.round(self.apply(coordinates), 6)


class NullPotential(PotentialFunction):
    name = "null"
    form = "V(x) = 0"

    def apply(self, coordinates: np.ndarray) -> float:
        return 0.0


class KronigPenneyPotential(PotentialFunction):
    name = "kronig-penney"
    form = "V(x) = V_0 if x <= 0.5 else -V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        return self.amplitude if x <= 0.5 else -self.amplitude


class SinePotential(PotentialFunction):
    name = "sine"
    form = "V(x) = V_0 sin(2 pi x)"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        return self.amplitude * np.sin(2 * np.pi * x)
