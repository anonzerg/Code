"""Initial Mass Function (IMF) models

Conventions:
    - Masses are in units of solar mass.
    - A model gives the unnormalised number of stars per unit mass.
"""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np
from scipy import integrate


MassFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class InitialMassFunction:
    name: str
    number_per_unit_mass: MassFunction


    def count_stars_in_mass_range(
            self,
            low: float,
            high: float,
            ) -> float:
        """Unnormalised number of stars with mass between `low` and `high`."""

        if low <= 0.0:
            raise ValueError(f'Mass must be positive. Got {low}.')

        if high <= low:
            return 0.0

        # Integrate in ln(M), where dN = (dN/dM) * M * dln(M).
        def number_per_log_mass(log_mass: float) -> float:
            mass = np.exp(log_mass)
            return float(self.number_per_unit_mass(mass)) * mass

        count, _ = integrate.quad(
                number_per_log_mass,
                np.log(low),
                np.log(high),
                limit=200,
                )

        return count


def broken_power_law(
        name: str,
        segments: list[tuple[float, float, float]],
        ) -> InitialMassFunction:
    """Continuous broken power law. `segments` are (low, high, alpha)."""

    for (_, previous_high, _), (low, _, _) in zip(segments[:-1], segments[1:]):
        if not np.isclose(previous_high, low):
            raise ValueError(f'{name}: segments must touch.')

    # Each amplitude makes its segment meet the previous one at the break mass.
    amplitudes = [1.0]
    for (_, _, previous_alpha), (mass_break, _, alpha) in zip(segments[:-1], segments[1:]):
        amplitudes.append(amplitudes[-1] * mass_break**(alpha-previous_alpha))

    def number_per_unit_mass(mass: np.ndarray) -> np.ndarray:
        mass = np.asarray(mass, dtype=float)
        density = np.zeros_like(mass)

        for amplitude, (low, high, alpha) in zip(amplitudes, segments):
            density = np.where(
                    (mass >= low) & (mass < high),
                    amplitude * mass**(-alpha),
                    density
                    )

        return density

    return InitialMassFunction(name, number_per_unit_mass)


def lognormal_power_law(
        name: str,
        mass_peak: float,
        sigma: float,
        tail_start: float,
        tail_alpha: float,
        ) -> InitialMassFunction:
    """Log-normal below `tail_start`, power law above it.

    `sigma` is the width in log10(M). The log-normal peaks at `mass_peak`.
    """

    # The log-normal is dN/dlogM and the power law is dN/dM = (dN/dlogM) / M.
    def lognormal(mass: np.ndarray) -> np.ndarray:
        return np.exp(-np.log10(mass / mass_peak)**2 / (2.0 * sigma**2))

    # The amplitude makes the two pieces meet at `tail_start`.
    tail_amplitude = lognormal(tail_start) * tail_start**(tail_alpha - 1.0)

    def number_per_unit_mass(mass: np.ndarray) -> np.ndarray:
        mass = np.asarray(mass, dtype=float)
        return np.where(
                mass < tail_start,
                lognormal(mass) / mass,
                tail_amplitude * mass**(-tail_alpha),
                )

    return InitialMassFunction(name, number_per_unit_mass)


MODELS: dict[str, InitialMassFunction] = {
        model.name: model
        for model in (
            broken_power_law(
                name='Salpeter-1955',
                segments=[(0.08, 100.0, 2.35)],
                ),

            broken_power_law(
                name='Kroupa-1993',
                segments=[
                    (0.08, 0.5, 1.3),
                    (0.5, 1.0, 2.2),
                    (1.0, 100.0, 2.7),
                    ],
                ),

            broken_power_law(
                name='Kroupa-2001',
                segments=[
                    (0.01, 0.08, 0.3),
                    (0.08, 0.5, 1.3),
                    (0.5, 100.0, 2.3),
                    ],
                ),

            lognormal_power_law(
                name='Chabrier-2003',
                mass_peak=0.08,
                sigma=0.69,
                tail_start=1.0,
                tail_alpha=2.3,
                ),

            lognormal_power_law(
                name='Chabrier-2005',
                mass_peak=0.2,
                sigma=0.55,
                tail_start=1.0,
                tail_alpha=2.35,
                ),
            )
        }
