#!/usr/bin/env python3

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


log = logging.getLogger(__name__)


# Constants
POPULATIONS: tuple[str, ...] = ('near', 'far')
REQUIRED_COLS: tuple[str, ...] = (
        'imf_name',
        'cluster_name',
        'population',
        'distance',
        'observed_white_dwarf_count',
        'expected_white_dwarf_count',
        )
DISP_NAMES: dict[str, str] = {
        'population': 'Population',
        'imf_name': 'IMF',
        'cluster_count': 'Clusters',
        'observed_total': 'Obs-WD',
        'expected_total': 'Exp-WD',
        'observed_over_expected': 'Obs/Exp',
        'chi_squared': 'chi2',
        'degrees_of_freedom': 'dof',
        'reduced_chi_squared': 'chi2/dof',
        'p_value': 'P-Value',
        'best': 'Best',
        }
FORMATTERS: dict[str, str] = {
        'Obs-WD': '{:.0f}'.format,
        'Exp-WD': '{:.1f}'.format,
        'Obs/Exp': '{:.2f}'.format,
        'chi2': '{:.1f}'.format,
        'chi2/dof': '{:.2f}'.format,
        'P-Value': '{:.3g}'.format,
        }
TABLE_NOTES: str = """\
Columns
    Clusters     Clusters used
    Observed WD  Total white dwarfs observed in clusters
    Expected WD  Total white dwarfs predicted by the IMF
    Obs/Exp      Observed WD / Expected WD
    chi2         Sum over clusters of (Obs - Exp)^2 / Exp
    dof          Number of clusters
    chi2/dof     Lower is a better match, about 1 means the scatter is Poisson-like
    P-Value      Chance of a chi2 at least this large if the IMF were correct
    Best         lowest chi2/dof inside the population
"""


def pearson_chi_squared(
        observed_counts: np.ndarray,
        expected_counts: np.ndarray,
        ) -> tuple[float, int, float]:

    chi_squared = float(
            np.sum((observed_counts - expected_counts)**2 / expected_counts)
            )
    degrees_of_freedom = len(observed_counts)
    p_value = float(stats.chi2.sf(chi_squared, degrees_of_freedom))

    return chi_squared, degrees_of_freedom, p_value


def select_clusters(
        results: pd.DataFrame,
        min_expected: float,
        ) -> pd.DataFrame:

    smallest_expected = results.groupby(
            ['population', 'cluster_name']
            ).expected_white_dwarf_count.transform('min')

    keep = (smallest_expected > 0.0) & (smallest_expected >= min_expected)

    dropped = results[~keep].groupby('population').cluster_name.nunique()
    for pop, count in dropped.items():
        log.info(
                f'In {pop} population, {count} clusters below {min_expected} expected white dwarfs.'
                )

    return results[keep]


def compare_imfs(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop in POPULATIONS:
        population_results = results[results.population == pop]

        if population_results.empty:
            log.warning(f'No {pop} clusters left. Skipping.')
            continue

        for imf_name, imf_results in population_results.groupby('imf_name', sort=False):
            observed_counts = imf_results.observed_white_dwarf_count.to_numpy(dtype=float)
            expected_counts = imf_results.expected_white_dwarf_count.to_numpy(dtype=float)

            chi_squared, degrees_of_freedom, p_value = pearson_chi_squared(
                    observed_counts,
                    expected_counts,
                    )

            rows.append(dict(
                population=pop,
                imf_name=imf_name,
                cluster_count=len(imf_results),
                observed_total=observed_counts.sum(),
                expected_total=expected_counts.sum(),
                observed_over_expected=observed_counts.sum() / expected_counts.sum(),
                chi_squared=chi_squared,
                degrees_of_freedom=degrees_of_freedom,
                reduced_chi_squared=chi_squared / degrees_of_freedom,
                p_value=p_value,
                ))

    table = pd.DataFrame(rows)

    lowest_in_pop = table.groupby(
            'population'
            ).reduced_chi_squared.transform('min')

    table['best'] = np.where(
            table.reduced_chi_squared == lowest_in_pop, '*', ''
            )

    return table


def describe_populations(results: pd.DataFrame) -> str:
    lines = []
    for pop in POPULATIONS:
        clusters = results[results.population == pop].drop_duplicates('cluster_name')

        if clusters.empty:
            continue

        lines.append(
                f'{pop}: {len(clusters)} clusters, '
                f'distance {clusters.distance.min():.0f} to {clusters.distance.max():.0f} pc'
                )

    return '\n'.join(lines)


def format_table(table: pd.DataFrame) -> str:
    disp = table.rename(columns=DISP_NAMES)
    lines = disp.to_string(index=False, formatters=FORMATTERS).splitlines()

    header, body = lines[0], lines[1:]

    formatted = [header]
    previous_pop = None
    for line, pop in zip(body, table.population):
        if previous_pop is not None and pop != previous_pop:
            formatted.append('')
        formatted.append(line)
        previous_pop = pop

    return '\n'.join(formatted)


def main(args: argparse.Namespace) -> None:
    try:
        results = pd.read_csv(args.input)
    except FileNotFoundError as e:
        log.error(e)
        sys.exit(1)

    missing = [col for col in REQUIRED_COLS if col not in results.columns]
    if missing:
        log.error(f'Missing columns {missing}. Run main.py again.')
        log.info(f'Available {list(results.columns)}')
        sys.exit(1)

    results = select_clusters(results, args.min_expected)
    if results.empty:
        log.error('No cluster left after the minimum expected count cut.')
        sys.exit(1)

    table = compare_imfs(results)

    report = '\n\n'.join((
        describe_populations(results),
        format_table(table),
        TABLE_NOTES,
        ))
    print(report)

    output = Path('Results') / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + '\n')
    table.to_csv(output.with_suffix('.csv'), index=False)

    log.info(f'Table written to {output} and {output.with_suffix(".csv")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'input',
            type=Path,
            help='csv file written by main.py',
            )
    parser.add_argument(
            '-o',
            '--output',
            type=Path,
            default=Path('imf_comparison.txt'),
            help='output file name inside Results/, (default is %(default)s)',
            )
    parser.add_argument(
            '-m',
            '--min-expected',
            type=float,
            default=1.0,
            help='drop clusters where any IMF expects fewer white dwarfs than this (default is %(default)s)',
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
