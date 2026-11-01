#!/usr/bin/env python3

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats


log = logging.getLogger(__name__)


# Constants
MASS_LIMITS: tuple[float, float] = (1.0, 8.0)      # In units of solar mass
FIT_MAX_ITER: int = 100
FIT_TOL: float = 1e-10
CONFIDENCE_LEVEL: float = 0.6827                   # 1 sigma. Use 0.95 for 95%
REQUIRED_COLS: tuple[str, ...] = (
        'turnoff_mass',
        'observed_white_dwarf_count',
        'observed_main_seq_count',
        'expected_ratio',
        'observed_ratio',
        )


def config_style(font_family: str, font_size: int) -> None:
    try:
        font_manager.findfont(font_family, fallback_to_default=False)
    except ValueError:
        log.warning(
                f'{font_family} font not found. Fallback to monospace.'
                )
        font_family = 'monospace'

    plt.rcParams.update({
        'font.family': font_family,
        'font.size': font_size,
        # 'axes.spines.top': False,
        # 'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.5,
        })


def fit_observed_curve(
        turnoff_mass: np.ndarray,
        white_dwarf_counts: np.ndarray,
        main_seq_counts: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:

    design_matrix = np.column_stack([np.ones_like(turnoff_mass), turnoff_mass])
    log_exposure = np.log(main_seq_counts)
    coeffs = np.array([
        np.log(white_dwarf_counts.sum()/main_seq_counts.sum()),
        0.0,
        ])
    for _ in range(FIT_MAX_ITER):
        linear_predictor = design_matrix @ coeffs + log_exposure
        predicted_count = np.exp(linear_predictor)
        working_response = (
                (linear_predictor-log_exposure)
                + (white_dwarf_counts-predicted_count) / predicted_count
                )
        weighted_design = design_matrix * predicted_count[:, None]
        updated_coeffs = np.linalg.solve(
                design_matrix.T @ weighted_design,
                weighted_design.T @ working_response,
                )
        converged = np.allclose(updated_coeffs, coeffs, rtol=0, atol=FIT_TOL)
        coeffs = updated_coeffs
        if converged:
            break
    else:
        log.warning('Fit did not converge.')

    predicted_count = np.exp(design_matrix @ coeffs+log_exposure)
    fisher_info = design_matrix.T @ (design_matrix * predicted_count[:, None])
    coeffs_covariance = np.linalg.inv(fisher_info)

    degree_of_freedom = len(white_dwarf_counts) - 2
    pearson_chi_squared = np.sum(
            (white_dwarf_counts - predicted_count)**2 / predicted_count
            )
    if degree_of_freedom > 0:
        dispersion = max(1.0, pearson_chi_squared / degree_of_freedom)
    else:
        dispersion = 1.0

    log.info(f'Dispersion {dispersion:.2f}')

    return coeffs, coeffs_covariance * dispersion


def poisson_confidence_interval(
        counts: np.ndarray,
        confidence_level: float = CONFIDENCE_LEVEL,
        ) -> tuple[np.ndarray, np.ndarray]:

    tail_prob = 1 - confidence_level
    lower_bound = np.where(
            counts > 0,
            stats.chi2.ppf(tail_prob / 2.0, 2.0 * np.maximum(counts, 1)) / 2.0,
            0.0,
            )
    upper_bound = stats.chi2.ppf(1.0 - tail_prob/2.0, 2.0*counts + 2.0) / 2.0
    return lower_bound, upper_bound


def plot_white_dwarf_main_seq_ratio_vs_turnoff_mass(
        table: pd.DataFrame,
        title: str
        ) -> plt.Figure:
    mass_low, mass_high = MASS_LIMITS
    visible = table[table.turnoff_mass.between(mass_low, mass_high)]
    hidden = len(table) - len(visible)
    if hidden:
        log.info(
                f'{hidden} clusters out of mass range. ({mass_low}, {mass_high})'
                )

    expected = visible.sort_values('turnoff_mass')
    clusters_with_white_dwarfs = visible[
            visible.observed_white_dwarf_count > 0
            ]
    clusters_without_white_dwarfs = visible[
            visible.observed_white_dwarf_count == 0
            ]
    log.info(
            f'{len(clusters_without_white_dwarfs)} clusters do not contain white dwarfs'
            )

    fig, axes = plt.subplots(figsize=(8, 8))

    axes.plot(
            expected.turnoff_mass,
            expected.expected_ratio,
            color='black',
            linestyle='--',
            label='Expected Curve (Kroupa IMF)',
            )

    if len(clusters_with_white_dwarfs):
        coeffs, coeffs_covariance = fit_observed_curve(
                visible.turnoff_mass.to_numpy(dtype=float),
                visible.observed_white_dwarf_count.to_numpy(dtype=float),
                visible.observed_main_seq_count.to_numpy(dtype=float),
                )

        intercept, slope = coeffs
        slope_std_err = np.sqrt(coeffs_covariance[1, 1])

        log.info(f'Observed fit: ln(white dwarfs / main seq) = {intercept:.4f} + {slope:.4f} * turnoff_mass')

        sigma_multiplier = stats.norm.ppf(0.5+CONFIDENCE_LEVEL / 2.0)
        mass_grid = np.linspace(visible.turnoff_mass.min(), visible.turnoff_mass.max(), 200)

        grid_design_matrix = np.column_stack([np.ones_like(mass_grid), mass_grid])
        log_ratio_fit = grid_design_matrix @ coeffs
        log_ratio_fit_std_err = np.sqrt(
                np.einsum(
                    'ij,jk,ik->i',
                    grid_design_matrix,
                    coeffs_covariance,
                    grid_design_matrix,
                    )
                )

        axes.plot(
                mass_grid,
                np.exp(log_ratio_fit),
                color='red',
                label='Observed (Fitted curve)',
                )
        axes.fill_between(
                mass_grid,
                np.exp(log_ratio_fit - log_ratio_fit_std_err*sigma_multiplier),
                np.exp(log_ratio_fit + log_ratio_fit_std_err*sigma_multiplier),
                color='red',
                alpha=0.2,
                linewidth=0.0,
                )

    white_dwarf_counts = clusters_with_white_dwarfs.observed_white_dwarf_count.to_numpy(dtype=float)
    main_seq_counts = clusters_with_white_dwarfs.observed_main_seq_count.to_numpy(dtype=float)
    count_lower_bound, count_upper_bound = poisson_confidence_interval(white_dwarf_counts)

    observed_ratio = white_dwarf_counts / main_seq_counts
    ratio_lower_err = observed_ratio - count_lower_bound/main_seq_counts
    ratio_upper_err = count_upper_bound/main_seq_counts - observed_ratio

    axes.errorbar(
            clusters_with_white_dwarfs.turnoff_mass,
            observed_ratio,
            yerr=[ratio_lower_err, ratio_upper_err],
            fmt='o',
            color='orange',
            markeredgecolor='gray',
            ecolor='gray',
            capsize=4.0,
            label=f'Observed {len(clusters_with_white_dwarfs)} clusters',
            )

    axes.set_xlim(mass_low, mass_high)
    axes.set_xlabel('Turnoff Mass')
    axes.set_ylabel('White Dwarfs / Main Seq')
    axes.set_title(title)
    fig.tight_layout()
    axes.legend(frameon=False)

    return fig



def main(args: argparse.Namespace) -> None:
    try:
        table = pd.read_csv(args.input)
    except FileNotFoundError as e:
        log.error(e)
        sys.exit(1)

    missing = [
            name for name in REQUIRED_COLS if name not in table.columns
            ]
    if missing:
        log.error(f'Missing columns {missing}. Available {list(table.columns)}')
        sys.exit(1)


    config_style(args.font_family, args.font_size)
    fig = plot_white_dwarf_main_seq_ratio_vs_turnoff_mass(table, title=args.title)

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
            default=Path('plot.png'),
            help='output plot path (default is Results/%(default)s)',
            )
    parser.add_argument(
            '-f',
            '--font-family',
            type=str,
            default='Iosevka',
            help='font family name (default is %(default)s)',
            )
    parser.add_argument(
            '-s',
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
