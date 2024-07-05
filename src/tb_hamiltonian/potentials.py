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
    elif type == "kronig-penney-break-even-sym":
        return KronigPenneyBreakEvenSymPotential()
    elif type == "kronig-penney-break-odd-sym":
        return KronigPenneyBreakOddSymPotential()
    elif type == "triangular":
        return TriangularPotential()
    elif type == "square":
        return SquarePotential()
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
        return f"{self.name} -> {self.form}"

    def __repr__(self) -> str:
        return self.__str__()

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


class KronigPenneyBreakEvenSymPotential(PotentialFunction):
    name = "kronig-penney-break-even-sym"
    form = "V(x) = V_0 if 0.25 <= x <= 0.5 else V_0 + 0.1*V_0 if 0.5 <= x <= 0.75 else V_0 - 0.1*V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        V = -self.amplitude
        perturb = 0.1 * self.amplitude
        start1, end1 = 0.25, 0.5
        start2, end2 = 0.5, 0.75

        if 0 <= x <= self.width:
            V = self.amplitude
        if start1 <= x <= end1:
            V += perturb
        if start2 <= x <= end2:
            V -= perturb

        return V


class KronigPenneyBreakOddSymPotential(PotentialFunction):
    name = "kronig-penney-break-odd-sym"
    form = "V(x) = V_0 if 0.125 <= x <= 0.375 else V_0 + 0.1*V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        V = -self.amplitude
        perturb = 0.1 * self.amplitude
        start1, end1 = 0.125, 0.375

        if 0 <= x <= self.width:
            V = self.amplitude
        if start1 <= x <= end1:
            V += perturb

        return V


class SinePotential(PotentialFunction):
    name = "sine"
    form = "V(x) = V_0 sin(2 pi x)"

    def apply(self, coordinates: np.ndarray) -> float:
        x, _, _ = coordinates
        return self.amplitude * np.sin(2 * np.pi * x)


class TriangularPotential(PotentialFunction):
    name = "triangular"
    form = "V(x, y) = V_0 * [cos(kx1*x) + cos(kx1/2*x + ky1*y) + cos(kx1/2*x - ky1*y)]"

    def apply(self, coordinates: np.ndarray) -> float:
        x, y, _ = coordinates
        kx1 = 2 * np.pi / self.width
        ky1 = 2 * np.pi / (self.width * np.sqrt(3.0))

        return self.amplitude * (np.cos(kx1 * x) +
                                 np.cos(kx1 / 2 * x + ky1 * y) +
                                 np.cos(kx1 / 2 * x - ky1 * y))


class SquarePotential(PotentialFunction):
    name = "square"
    form = "V(x, y) = V_0 * [sin(k*x)^2 + sin(k*y)^2]"

    def apply(self, coordinates: np.ndarray) -> float:
        x, y, _ = coordinates
        k = 2 * np.pi / self.width
        return self.amplitude * (np.sin(k * x)**2 + np.sin(k * y)**2)
