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
    params: dict = {}

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
        amplitude = self.params.get("amplitude", 1.0)
        x, _, _ = coordinates
        return amplitude if x <= 0.5 else -amplitude


class KronigPenneyBreakEvenSymPotential(PotentialFunction):
    name = "kronig-penney-break-even-sym"
    form = "V(x) = V_0 if start1 <= x <= end1 else V_0 + beta*V_0 if start2 <= x <= end2 else V_0 - beta*V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        width = self.params.get("width", 0.5)
        beta = self.params.get("beta", 0.1)
        start1 = self.params.get("start1", 0.25)
        end1 = self.params.get("end1", 0.5)
        start2 = self.params.get("start2", 0.5)
        end2 = self.params.get("end2", 0.75)
        x, _, _ = coordinates
        V = -amplitude
        perturb = beta * amplitude

        if 0 <= x <= width:
            V = amplitude
        if start1 <= x <= end1:
            V += perturb
        if start2 <= x <= end2:
            V -= perturb

        return V


class KronigPenneyBreakOddSymPotential(PotentialFunction):
    name = "kronig-penney-break-odd-sym"
    form = "V(x) = V_0 if start <= x <= end else V_0 + beta*V_0"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        width = self.params.get("width", 0.5)
        beta = self.params.get("beta", 0.1)
        start = self.params.get("start1", 0.125)
        end = self.params.get("end1", 0.375)

        x, _, _ = coordinates
        V = -amplitude
        perturb = beta * amplitude

        if 0 <= x <= width:
            V = amplitude
        if start <= x <= end:
            V += perturb

        return V


class SinePotential(PotentialFunction):
    name = "sine"
    form = "V(x) = V_0 * sin(2pi*n*x)"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        n = self.params.get("n", 1)
        x, _, _ = coordinates
        return amplitude * np.sin(2 * np.pi * n * x)


class TriangularPotential(PotentialFunction):
    name = "triangular"
    form = "V(x, y) = V_0 * [cos(kx1*x) + cos(kx1/2*x + ky1*y) + cos(kx1/2*x - ky1*y)]"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        width = self.params.get("width", 1.0)
        height = self.params.get("height", 2 * width)
        n = self.params.get("n", 1)
        m = self.params.get("m", 1)
        x, y, _ = coordinates
        kx1 = 2 * np.pi * n / width
        ky1 = 2 * np.pi * m / height

        return amplitude * (
            np.cos(kx1 * x) + np.cos(kx1 / 2 * x + ky1 * y) + np.cos(kx1 / 2 * x - ky1 * y)
        )


class SquarePotential(PotentialFunction):
    name = "square"
    form = "V(x, y) = V_0 * [sin(k*x)^2 + sin(k*y)^2]"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        width = self.params.get("width", 1.0)
        x, y, _ = coordinates
        k = 2 * np.pi / width
        return amplitude * (np.sin(k * x) ** 2 + np.sin(k * y) ** 2)
