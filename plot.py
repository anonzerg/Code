#!/usr/bin/env python3

from __future__ import annotations

import sys
import logging
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats


log = logging.getLogger(__name__)


# Constants
TURNOFF_MASS_LIMITS: tuple[float, float] = (1.0, 8.0)  # In units of solar mass
FIT_MAX_ITER: int = 100
FIT_TOLERANCE: float = 1e-10
REQUIRED_COLS: tuple[str, ...] = (
        'turnoff_mass',
        'observed_white_dwarf_count',
        'observed_main_seq_count',
        'expected_ratio',
        )


def config_style(font_family: str, font_size: int) -> None:
    try:
        font_manager.findfont(font_family, fallback_to_default=False)
    except ValueError:
        log.warning(f'{font_family} font not found. Fallback to `monospace`.')
        font_family = 'monospace'

    plt.rcParams.update({
        'font.family': font_family,
        'font.size': font_size,
        'axes.grid': True,
        'grid.alpha': 0.5,
        })


def fit_poisson(
        turnoff_mass: np.ndarray,
        white_dwarf_counts: np.ndarray,
        main_seq_counts: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:

    design_matrix = np.column_stack([np.ones_like(turnoff_mass), turnoff_mass])
    log_exposure = np.log(main_seq_counts)
    coefficients = np.array([
        np.log(white_dwarf_counts.sum()/main_seq_counts.sum()),
        0.0,
        ])
    for _ in range(FIT_MAX_ITER):
        linear_predictor = design_matrix @ coefficients + log_exposure
        predicted_count = np.exp(linear_predictor)
        working_response = (
                (linear_predictor-log_exposure)
                + (white_dwarf_counts-predicted_count) / predicted_count
                )
        weighted_design = design_matrix * predicted_count[:, None]
        updated_coefficients = np.linalg.solve(
                design_matrix.T @ weighted_design,
                weighted_design.T @ working_response,
                )
        converged = np.allclose(
                updated_coefficients,
                coefficients,
                rtol=0,
                atol=FIT_TOLERANCE
                )
        coefficients = updated_coefficients
        if converged:
            break
    else:
        log.warning('Fit did not converge.')

    predicted_count = np.exp(design_matrix @ coefficients+log_exposure)
    fisher_info = design_matrix.T @ (design_matrix * predicted_count[:, None])
    covariance = np.linalg.inv(fisher_info)

    degrees_of_freedom = len(white_dwarf_counts) - 2
    pearson_chi_squared = np.sum(
            (white_dwarf_counts - predicted_count)**2 / predicted_count
            )
    if degrees_of_freedom > 0:
        dispersion = max(1.0, pearson_chi_squared / degrees_of_freedom)
    else:
        dispersion = 1.0

    log.info(f'Dispersion {dispersion:.2f}')

    return coefficients, covariance * dispersion


def poisson_confidence_interval(
        counts: np.ndarray,
        confidence_level: float,
        ) -> tuple[np.ndarray, np.ndarray]:

    tail_prob = 1 - confidence_level
    lower_bound = np.where(
            counts > 0,
            stats.chi2.ppf(tail_prob / 2.0, 2.0 * np.maximum(counts, 1)) / 2.0,
            0.0,
            )
    upper_bound = stats.chi2.ppf(1.0 - tail_prob/2.0, 2.0*counts + 2.0) / 2.0
    return lower_bound, upper_bound


def plot_ratio_vs_turnoff_mass(
        clusters: pd.DataFrame,
        confidence_level: float,
        title: str
        ) -> plt.Figure:
    mass_min, mass_max = TURNOFF_MASS_LIMITS
    visible = clusters[clusters.turnoff_mass.between(mass_min, mass_max)]
    hidden = len(clusters) - len(visible)
    if hidden:
        log.info(f'{hidden} clusters out of mass range.')
        log.info(f'Mass range is ({mass_min}, {mass_max})')

    expected = visible.sort_values('turnoff_mass')
    with_white_dwarfs = visible[
            visible.observed_white_dwarf_count > 0
            ]
    without_white_dwarfs = visible[
            visible.observed_white_dwarf_count == 0
            ]
    log.info(
            f'{len(without_white_dwarfs)} clusters do not contain white dwarfs'
            )

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
            expected.turnoff_mass,
            expected.expected_ratio,
            color='black',
            linestyle='--',
            label='Expected Curve (Kroupa IMF)',
            )

    if len(with_white_dwarfs) > 2:
        coefficients, covariance = fit_poisson(
                with_white_dwarfs.turnoff_mass.to_numpy(dtype=float),
                with_white_dwarfs.observed_white_dwarf_count.to_numpy(dtype=float),
                with_white_dwarfs.observed_main_seq_count.to_numpy(dtype=float),
                )

        intercept, slope = coefficients
        slope_std_err = np.sqrt(covariance[1, 1])

        log.info(
                'Observed fit: ln(white dwarfs / main seq) = '
                f'{intercept:.4f} + {slope:.4f} * turnoff_mass'
                )

        z_score = stats.norm.ppf(0.5 + confidence_level/2.0)
        mass_grid = np.linspace(
                with_white_dwarfs.turnoff_mass.min(),
                with_white_dwarfs.turnoff_mass.max(),
                200
                )

        grid_design_matrix = np.column_stack([np.ones_like(mass_grid), mass_grid])
        log_ratio_fit = grid_design_matrix @ coefficients
        log_ratio_fit_std_err = np.sqrt(
                np.einsum(
                    'ij,jk,ik->i',
                    grid_design_matrix,
                    covariance,
                    grid_design_matrix,
                    )
                )

        ax.plot(
                mass_grid,
                np.exp(log_ratio_fit),
                color='red',
                label='Observed (Fitted curve)',
                )
        ax.fill_between(
                mass_grid,
                np.exp(log_ratio_fit - log_ratio_fit_std_err*z_score),
                np.exp(log_ratio_fit + log_ratio_fit_std_err*z_score),
                color='red',
                alpha=0.2,
                linewidth=0.0,
                )

    white_dwarf_counts = with_white_dwarfs.observed_white_dwarf_count.to_numpy(dtype=float)
    main_seq_counts = with_white_dwarfs.observed_main_seq_count.to_numpy(dtype=float)
    count_lower, count_upper = poisson_confidence_interval(
            white_dwarf_counts,
            confidence_level,
            )

    observed_ratio = white_dwarf_counts / main_seq_counts
    ratio_lower = observed_ratio - count_lower/main_seq_counts
    ratio_upper = count_upper/main_seq_counts - observed_ratio

    ax.errorbar(
            with_white_dwarfs.turnoff_mass,
            observed_ratio,
            yerr=[ratio_lower, ratio_upper],
            fmt='o',
            color='orange',
            markeredgecolor='gray',
            ecolor='gray',
            capsize=4.0,
            label=f'Observed {len(with_white_dwarfs)} clusters',
            )

    ax.set_xlim(mass_min, mass_max)
    ax.set_xlabel('Turnoff Mass')
    ax.set_ylabel('White Dwarfs / Main Seq')
    ax.set_title(title)
    fig.tight_layout()
    ax.legend(loc='upper right', frameon=False)

    return fig



def main(args: argparse.Namespace) -> None:
    try:
        clusters = pd.read_csv(args.input)
    except FileNotFoundError as e:
        log.error(e)
        sys.exit(1)

    missing = [
            col for col in REQUIRED_COLS if col not in clusters.columns
            ]
    if missing:
        log.error(f'Missing columns {missing}.')
        log.info(f'Available {list(clusters.columns)}')
        sys.exit(1)

    confidence_level = math.erf(args.sigma / math.sqrt(2))

    config_style(args.font_family, args.font_size)
    fig = plot_ratio_vs_turnoff_mass(
            clusters,
            title=args.title,
            confidence_level=confidence_level,
            )

    args.output = 'Results' / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    
    log.info(f'Plot created at {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'input',
            type=Path,
            help='input csv file',
            )
    parser.add_argument(
            '-o',
            '--output',
            type=Path,
            default=Path('plot.svg'),
            help='output file name inside Results/, (default is %(default)s)',
            )
    parser.add_argument(
            '-f',
            '--font-family',
            type=str,
            default='Iosevka',
            help='font family name, falls back to monospace if not found (default is %(default)s)',
            )
    parser.add_argument(
            '-z',
            '--font-size',
            type=int,
            default=16,
            help='font size (default is %(default)s)',
            )
    parser.add_argument(
            '-d',
            '--dpi',
            type=int,
            default=300,
            help='resolution (default is %(default)s)',
            )
    parser.add_argument(
            '-t',
            '--title',
            type=str,
            default='title',
            help='plot title (default is %(default)s)',
            )
    parser.add_argument(
            '-s',
            '--sigma',
            type=float,
            default=1.0,
            help='confidence level in standard deviations, used for error bars and fit band (default is %(default)s)',
            )
    args = parser.parse_args()

    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            )
    try:
        main(args)
    except KeyboardInterrupt:
        sys.exit('Interrupted!')
