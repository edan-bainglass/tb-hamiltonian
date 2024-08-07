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
    elif type == "rectangular":
        return RectangularPotential()
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
    form = "V(x, y) = V_0 * [cos(Q1*r) + cos(Q2*r) + cos(Q3*r) + cos(Q4*r) + cos(Q5*r) + cos(Q6*r)]"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        Q0 = self.params.get("Q0", 1)
        r = coordinates[:2]

        def Q(n):
            arg = 2 * np.pi * n / 6
            return Q0 * np.array([np.cos(arg), np.sin(arg)])

        return amplitude * sum(np.cos(Q(n) @ r) for n in range(1, 7))


class RectangularPotential(PotentialFunction):
    name = "rectangular"
    form = "V(x, y) = V_0 * [sin(kx*x)^2 + sin(ky*y)^2]"

    def apply(self, coordinates: np.ndarray) -> float:
        amplitude = self.params.get("amplitude", 1.0)
        width = self.params.get("width", 1.0)
        height = self.params.get("height", width)
        x, y, _ = coordinates
        kx = 2 * np.pi / width
        ky = 2 * np.pi / height
        return amplitude * (np.sin(kx * x) ** 2 + np.sin(ky * y) ** 2)
