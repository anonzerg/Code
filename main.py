#!/usr/bin/env python3

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path

import ezpadova
import numpy as np
import pandas as pd

from imf import InitialMassFunction, MODELS


log = logging.getLogger(__name__)


# Constants
LOWEST_MAIN_SEQ_MASS: float = 0.08          # In units of solar mass.
MAX_WHITE_DWARF_PROG_MASS: float = 8.0      # Use `12.0` for loose max value.
MIN_STARS_IN_TOP_THIRD: int = 10
MAX_MAIN_SEQ_PARSEC_LABEL: int = 1          # PARSEC label: 0 = pre-MS, 1 = MS, ...
ISOCHRONE_LOG_AGE: tuple[float, float, float] = (6.4, 10.1, 0.05)
ISOCHRONE_Z: tuple[float, float, float] = (0.015, 0.015, 0.0)
PARSEC_TRACK: str = 'parsec_CAF09_v1.2S'

CLUSTER_COLS: dict[str, str] = {
        # Hunt & Reffert 2024 clusters data.
        'Name': 'cluster_name',
        'Type': 'cluster_type',
        'dist50': 'dist',
        'logAge50': 'log_age',
        }

MEMBER_COLS: dict[str, str] = {
        # Hunt & Reffert 2024 members data.
        'Name': 'cluster_name',
        'GaiaDR3': 'gaia_id',
        'Mass50': 'mass',
        'RUWE': 'ruwe',
        }

WHITE_DWARF_COLS: dict[str, str] = {
        # Gentile Fusillo et. al. 2021 white dwarfs data.
        'GaiaEDR3': 'gaia_id',
        'Pwd': 'prob',
        }


def load_table(path: str | Path, columns: dict[str, str]) -> pd.DataFrame:
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == '.parquet':
        table = pd.read_parquet(path)
        missing = [col for col in columns if col not in table.columns]
        if missing:
            raise ValueError(
                    f'Missing columns {missing}\n'
                    f'Path is {path}\n'
                    f'Available {list(table.columns)}'
                    )

    else:
        raise ValueError(f'{suffix} files are not supported.')

    return table[list(columns)].rename(columns=columns)


def load_isochrones(path: str | Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        path = Path(path)

    if path.exists():
        log.info(f'Isochrones already cached. using {path}')
        return pd.read_parquet(path)

    isochrones = ezpadova.get_isochrones(
            logage=ISOCHRONE_LOG_AGE,
            Z=ISOCHRONE_Z,
            track_parsec=PARSEC_TRACK,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    isochrones.to_parquet(path, index=False)
    return isochrones


def build_mass_table(isochrones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for log_age, isochrone in isochrones.groupby('logAge'):
        rows.append(dict(
            log_age=log_age,
            turnoff_mass=isochrone.loc[
                isochrone.label <= MAX_MAIN_SEQ_PARSEC_LABEL,
                'Mini'
                ].max(),
            white_dwarf_prog_mass=isochrone.Mini.max(),
            ))

    return pd.DataFrame(rows).sort_values('log_age').reset_index(drop=True)


def select_calibration_stars(
        masses: np.ndarray,
        turnoff_mass: float,
        ) -> tuple[np.ndarray, float] | None:

    main_seq_masses = np.sort(masses[masses < turnoff_mass])[::-1]

    top_third = len(main_seq_masses) // 3
    if top_third < MIN_STARS_IN_TOP_THIRD:
        return None

    calibration_min_mass = main_seq_masses[top_third - 1]

    return main_seq_masses, calibration_min_mass


def counts_from_imf(
        main_seq_masses: np.ndarray,
        calibration_min_mass: float,
        turnoff_mass: float,
        white_dwarf_prog_mass: float,
        imf: InitialMassFunction,
        ) -> dict[str, float]:

    calibration_stars = int((main_seq_masses >= calibration_min_mass).sum())
    imf_scale = calibration_stars / imf.count_stars_in_mass_range(
            calibration_min_mass,
            turnoff_mass,
            )

    return dict(
            expected_white_dwarf_count = imf_scale * imf.count_stars_in_mass_range(
                white_dwarf_prog_mass,
                MAX_WHITE_DWARF_PROG_MASS,
                ),
            expected_main_seq_count = imf_scale * imf.count_stars_in_mass_range(
                LOWEST_MAIN_SEQ_MASS,
                turnoff_mass
                ),
            )


def main(args: argparse.Namespace) -> None:
    try:
        clusters = load_table('./Tables/clusters.parquet', CLUSTER_COLS)
        members = load_table('./Tables/members.parquet', MEMBER_COLS)
        white_dwarfs = load_table('./Tables/maincat.parquet', WHITE_DWARF_COLS)
        isochrones = load_isochrones('./Tables/isochrones.parquet')
    except Exception:
        log.exception('Could not load input tables.')
        sys.exit(1)

    missing = [col for col in ('logAge', 'Mini', 'label') if col not in isochrones.columns]
    if missing:
        log.error(
                f'Missing isochrone columns {missing}\n'
                f'Available {list(isochrones.columns)}'
                )
        sys.exit(1)

    mass_table = build_mass_table(isochrones).dropna().reset_index(drop=True)
    log_age_low = mass_table.log_age.iloc[0]
    log_age_high = mass_table.log_age.iloc[-1]

    clusters = clusters[
            (clusters.cluster_type == 'o')
            & (clusters.dist < args.dist)
            ]

    members = members[
            (members.cluster_name.isin(clusters.cluster_name))
            & (members.ruwe < args.ruwe)
            ]

    white_dwarf_ids = set(
            white_dwarfs.gaia_id[white_dwarfs.prob > args.prob]
            )

    members = members.assign(
            is_white_dwarf = members.gaia_id.isin(white_dwarf_ids)
            )

    if not members.is_white_dwarf.any():
        log.warning(
                'No cluster member matched the White Dwarfs catalog.\n'
                'Check Gaia ID dtypes.'
                )

    members_by_cluster = dict(tuple(members.groupby('cluster_name')))

    results = []
    for cluster_name, log_age in zip(clusters.cluster_name, clusters.log_age):
        if cluster_name not in members_by_cluster:
            continue

        if not np.isfinite(log_age):
            log.warning(f'Skipping {cluster_name}. Missing logAge.')
            continue

        if not log_age_low <= log_age <= log_age_high:
            log.warning(f'Skipping {cluster_name}. {log_age} outside isochrone grid.')
            continue

        cluster_members = members_by_cluster[cluster_name]

        turnoff_mass = np.interp(
                log_age,
                mass_table.log_age,
                mass_table.turnoff_mass
                )
        white_dwarf_prog_mass = np.interp(
                log_age, 
                mass_table.log_age, 
                mass_table.white_dwarf_prog_mass,
                )
        
        calibration = select_calibration_stars(
                cluster_members.mass.dropna().to_numpy(),
                turnoff_mass,
                )

        if calibration is None:
            log.info(f'Skipping {cluster_name}. Too few calibration stars.')
            continue

        main_seq_masses, calibration_min_mass = calibration
        observed_white_dwarf_count = int(cluster_members.is_white_dwarf.sum())

        for imf in MODELS.values():
            expected_counts = counts_from_imf(
                    main_seq_masses,
                    calibration_min_mass,
                    turnoff_mass,
                    white_dwarf_prog_mass,
                    imf,
                    )
        
            results.append(dict(
                imf_name=imf.name,
                cluster_name=cluster_name,
                turnoff_mass=turnoff_mass,
                white_dwarf_prog_mass=white_dwarf_prog_mass,
                observed_white_dwarf_count=observed_white_dwarf_count,
                observed_main_seq_count=len(main_seq_masses),
                **expected_counts,
                ))

    if not results:
        log.error('No cluster passed the selection process.')
        sys.exit(1)

    results = pd.DataFrame(results)
    results['expected_ratio'] = results.expected_white_dwarf_count / results.expected_main_seq_count
    results['observed_ratio'] = results.observed_white_dwarf_count / results.observed_main_seq_count

    output = Path('Results') / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    log.info(f'Results written to {output}')

    log.info(f'{results.cluster_name.nunique()} clusters passed the selection process.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-o',
            '--output',
            type=Path,
            default=Path('white_dwarf_expected_vs_observed.csv'),
            help='output csv path (default is Results/%(default)s)',
            )
    parser.add_argument(
            '-d',
            '--dist',
            type=float,
            default=500.0,
            help='max distance in parsec (default is %(default)s pc)',
            )
    parser.add_argument(
            '-e',
            '--ruwe',
            type=float,
            default=1.4,
            help='renormalized unit weight error (default is %(default)s)',
            )
    parser.add_argument(
            '-p',
            '--prob',
            type=float,
            default=0.75,
            help='probability being a white dwarfs (default is %(default)s)',
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
