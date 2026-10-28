#!/usr/bin/env python3

import sys
import sqlite3
import logging
import argparse
from pathlib import Path

import ezpadova
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


log = logging.getLogger(__name__)


# Constants
LOWEST_MS_MASS: float = 0.08                # In units of solar mass.
MAX_WHITE_DWARF_PROG_MASS: float = 8.0      # Use `12.0` for loose max value
MIN_WHITE_DWARF_PROB: float = 0.75
MAX_DIST: float = 1000                      # In units of pc.
MIN_STARS_IN_TOP_THIRD: int = 3
MAX_MAIN_SEQ_PARSEC_LABEL: int = 1          # PARSEC label: 0 = pre-MS, 1 = MS, ...
ISOCHRONE_LOG_AGE: tuple[float, float, float] = (6.4, 10.1, 0.05)
ISOCHRONE_Z: tuple[float, float, float] = (0.015, 0.015, 0.0)
PARSEC_TRACK: str = 'parsec_CAF09_v1'

COLS: dict[str, str] = dict(
        # Hunt & Reffert 2024 clusters data.
        cluster_name='Name',
        cluster_type='Type',
        cluster_dist='dist50',
        cluster_log_age='logAge',

        # Hunt & Reffert 2024 members data.
        member_cluster_name='Name',
        member_gaia_id='GaiaDR3',
        member_mass='Mass',

        # Gentile Fusillo et. al. 2021 white dwarfs data.
        white_dwarf_gaia_id='GaiaEDR3',
        white_dwarf_prob='Pwd',
        )

# For each segment (low, high, alpha)
IMF_SEGMENTS: list[tuple[float, float, float]] = [
        (0.01, 0.08, 0.3),          # Brown dwarfs (kept for IFM continuity)
        (0.08, 0.50, 1.3),          # Low-mass stars
        (0.50, 100.0, 2.3),         # Intermediate-mass stars
        ]

IMF_AMPS: list[float] = [1.0]
for (_, _, previous_alpha), (mass_break, _, alpha) in zip(IMF_SEGMENTS[:-1], IMF_SEGMENTS[1:]):
    IMF_AMPS.append(IMF_AMPS[-1] * mass_break**(alpha-previous_alpha))


def count_stars_in_mass_range(mass_low: float, mass_high: float) -> float:
    if mass_high < mass_low:
        return 0.0

    total = 0.0
    for amplitude, (segment_low, segment_high, alpha) in zip(IMF_AMPS, IMF_SEGMENTS):
        overlap_low = max(mass_low, segment_low)
        overlap_high = min(mass_high, segment_high)

        if overlap_high > overlap_low:
            beta = 1.0 - alpha
            total += (amplitude/beta) * (overlap_high**beta - overlap_low**beta)

    return total
        

def load_table(path: str | Path) -> pd.DataFrame:
    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix == '.parquet':
        return pd.read_parquet(path)

    elif path.suffix in ('.sqlite', '.sqlite3', '.db', '.db3'):
        con = sqlite3.connect(path)
        dataframe = pd.read_sql(f'SELECT * FROM {path.stem}', con)
        con.close()
        return dataframe
    else:
        raise ValueError(f'{path.suffix} files are not supported.')


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


def estimate_white_dwarfs_main_seq(
        member_masses: np.ndarray,
        turnoff_mass: float,
        white_dwarf_prog_mass: float,
        ) -> dict[str, float] | None:

    main_seq_masses = np.sort(member_masses[member_masses < turnoff_mass])[::-1]

    top_third = len(main_seq_masses) // 3
    if top_third < MIN_STARS_IN_TOP_THIRD:
        return None

    faintest_top_third_mass = main_seq_masses[top_third - 1]
    stars_in_calibrated_range = int(
            (main_seq_masses >= faintest_top_third_mass).sum()
            )
    imf_scale = stars_in_calibrated_range / count_stars_in_mass_range(
            faintest_top_third_mass,
            turnoff_mass
            )

    return dict(
            expected_white_dwarf_count = imf_scale * count_stars_in_mass_range(
                white_dwarf_prog_mass,
                MAX_WHITE_DWARF_PROG_MASS,
                ),
            estimated_main_seq_count = imf_scale * count_stars_in_mass_range(
                LOWEST_MS_MASS, turnoff_mass),
            )


def main(args: argparse.Namespace) -> None:
    try:
        clusters = load_table('./Tables/clusters.sqlite')
        members = load_table('./Tables/members.parquet')
        white_dwarfs = load_table('./Tables/maincat.parquet')
        isochrones = ezpadova.get_isochrones(
                logage=ISOCHRONE_LOG_AGE,
                Z=ISOCHRONE_Z,
                track_parsec=PARSEC_TRACK,
                )
    except Exception as e:
        log.error(e)
        sys.exit(1)

    for table, keys in [
            (clusters, ['cluster_name', 'cluster_type', 'cluster_dist', 'cluster_log_age']),
            (members, ['member_cluster_name', 'member_gaia_id', 'member_mass']),
            (white_dwarfs, ['white_dwarf_gaia_id', 'white_dwarf_prob']),
            ]:
        missing = [COLS[key] for key in keys if COLS[key] not in table.columns]

        if missing:
            log.error(f'Missing columns {missing}. Available {list(table.columns)}')
            sys.exit(1)

    missing = [name for name in ('logAge', 'Mini', 'label') if name not in isochrones.columns]
    if missing:
        log.error(f'Missing isochrone columns {missing}. Available {list(isochrones.columns)}')
        sys.exit(1)

    mass_table = build_mass_table(isochrones)

    clusters = clusters[
            (clusters[COLS['cluster_type']] == 'o')
            & (clusters[COLS['cluster_dist']] < MAX_DIST)
            ]

    members = members[
            members[COLS['member_cluster_name']].isin(clusters[COLS['cluster_name']])
            ]

    white_dwarf_ids = set(white_dwarfs.loc[
        white_dwarfs[COLS['white_dwarf_prob']] > MIN_WHITE_DWARF_PROB,
        COLS['white_dwarf_gaia_id']
        ])

    members = members.assign(
            is_white_dwarf = members[COLS['member_gaia_id']].isin(white_dwarf_ids)
            )

    members_by_cluster = dict(tuple(members.groupby(COLS['member_cluster_name'])))

    results = []
    for cluster_name, log_age in zip(clusters[COLS['cluster_name']], clusters[COLS['cluster_log_age']]):
        if cluster_name not in members_by_cluster:
            continue

        if not np.isfinite(log_age):
            log.warning(f'Skipping {cluster_name}. Missing logAge.')
            continue

        log_age_grid_low = mass_table.log_age.iloc[0]
        log_age_grid_high = mass_table.log_age.iloc[-1]

        if not log_age_grid_low <= log_age <= log_age_grid_high:
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
        
        counts = estimate_white_dwarfs_main_seq(
                cluster_members[COLS['member_mass']].dropna().to_numpy(),
                turnoff_mass,
                white_dwarf_prog_mass
                )
        if counts is None:
            continue

        results.append(dict(
            cluster_name=cluster_name,
            turnoff_mass=turnoff_mass,
            white_dwarf_prog_mass=white_dwarf_prog_mass,
            observed_white_dwarfs=int(cluster_members.is_white_dwarf.sum()),
            **counts,
            ))

    if not results:
        log.error('No cluster passed the selection process.')
        sys.exit(1)

    results = pd.DataFrame(results)
    results['observed_ratio'] = results.observed_white_dwarfs / results.estimated_main_seq_count
    results.to_csv(args.output, index=False)
    log.info(f'{len(results)} clusters | expected White Dwarfs {results.expected_white_dwarf_count.sum():.0f} '
             f'| observed White Dwarfs {results.observed_white_dwarfs.sum()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-o',
            '--output',
            type=Path,
            default=Path('white_dwarf_expected_vs_observed.csv'),
            help='output csv path (default is %(default)s',
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
        log.error('Interrupted!')
        sys.exit(1)
