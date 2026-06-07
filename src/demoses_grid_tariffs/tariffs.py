"""Grid-tariff configuration shared by the district heating and prosumer models.

This module defines a small, direction-aware representation of grid tariffs so that
both the DHN optimization model and the substation-prosumer optimization model can
apply capacity and/or volumetric tariffs on withdrawal and/or injection in a
consistent way.

Two delivery mechanisms are supported and merged here:
  1. A ``tariffs`` block in the workflow YAML config.
  2. Command-line overrides (direction-suffixed flags), which take precedence.

Conventions
-----------
* Volumetric tariffs are in EUR/MWh and given as a snapshot-indexed time series.
* Capacity tariffs use a single level in EUR/MW-month and an optional snapshot-indexed
  hourly weight profile in [0, 1]. The weight is applied to each snapshot *before*
  the monthly maximum is taken (the regulator's "weighted kW max" / kWmax_gewogen).
  A missing weight profile means an unweighted (all-ones) capacity tariff.
* Injection tariffs default to ``None`` (not applied) unless explicitly configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DIRECTIONS = ("withdrawal", "injection")


@dataclass
class DirectionTariff:
    """Tariffs that apply in a single direction (withdrawal or injection).

    Attributes:
    -----------
        volumetric: Snapshot-indexed volumetric tariff [EUR/MWh], or None.
        cap_level: Capacity tariff level [EUR/MW-month], or None.
        cap_weights: Snapshot-indexed hourly weights in [0, 1] applied inside the
            monthly maximum (weighted kW max). None means unweighted (all ones).
    """

    volumetric: pd.Series | None = None
    cap_level: float | None = None
    cap_weights: pd.Series | None = None

    @property
    def has_volumetric(self) -> bool:
        return self.volumetric is not None

    @property
    def has_capacity(self) -> bool:
        return self.cap_level is not None


@dataclass
class Tariffs:
    """Container for the withdrawal- and injection-side tariffs."""

    withdrawal: DirectionTariff = field(default_factory=DirectionTariff)
    injection: DirectionTariff = field(default_factory=DirectionTariff)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.withdrawal.has_volumetric,
                self.withdrawal.has_capacity,
                self.injection.has_volumetric,
                self.injection.has_capacity,
            ]
        )


def _load_snapshot_series(path: str | Path, snapshots: pd.DatetimeIndex, label: str) -> pd.Series:
    """Load a single-value-column, snapshot-indexed CSV and align it to ``snapshots``.

    Args:
    -----
        path: Path to the CSV file. Must have a datetime index in the first column.
        snapshots: The model snapshots the series must cover.
        label: Human-readable label used in error/log messages.

    Returns:
    --------
        A pandas Series aligned to ``snapshots``.

    Raises:
    -------
        ValueError: If the file is shorter than ``snapshots`` (i.e. would be truncated)
            or contains NaN values over the modeled snapshots.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if len(df) < len(snapshots):
        raise ValueError(
            f"{label} tariff file '{path}' has {len(df)} rows < {len(snapshots)} snapshots. "
            "Provide a profile that fully covers the modeled snapshots (no truncation)."
        )

    # Pick the value column: prefer well-known names, otherwise the first column.
    if "vol_tou_tariffs" in df.columns:
        series = df["vol_tou_tariffs"].copy()
    elif "value" in df.columns:
        series = df["value"].copy()
    else:
        series = df.iloc[:, 0].copy()

    series = series.iloc[: len(snapshots)]
    series.index = snapshots

    if series.isnull().any():
        raise ValueError(f"{label} tariff file '{path}' contains NaN values over the modeled snapshots.")

    return series


def _resolve_path(path, base_dir):
    """Resolve a tariff file path. Relative paths are looked up inside ``base_dir`` first.

    This lets scenario configs list bare filenames (e.g. ``weights_withdrawal.csv``) that live in the
    scenario's ``results/<stem>/inputs/tariff_params/`` folder, while still accepting absolute paths
    or explicit relative paths (e.g. CLI overrides like ``data/...``) unchanged.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / p
        if candidate.exists():
            return candidate
    return p


def _build_direction(
    yaml_block: dict,
    snapshots: pd.DatetimeIndex,
    vol_override: str | Path | None,
    cap_level_override: float | None,
    cap_weights_override: str | Path | None,
    direction: str,
    base_dir: str | Path | None = None,
) -> DirectionTariff:
    """Build a DirectionTariff from a YAML sub-block plus optional CLI overrides."""
    yaml_block = yaml_block or {}
    capacity_block = yaml_block.get("capacity", {}) or {}

    # Volumetric (CLI overrides YAML).
    vol_path = _resolve_path(vol_override if vol_override is not None else yaml_block.get("volumetric"), base_dir)
    volumetric = (
        _load_snapshot_series(vol_path, snapshots, f"{direction} volumetric") if vol_path is not None else None
    )

    # Capacity level (CLI overrides YAML).
    cap_level = cap_level_override if cap_level_override is not None else capacity_block.get("level")
    cap_level = float(cap_level) if cap_level is not None else None

    # Capacity weights (CLI overrides YAML). None => unweighted (all ones).
    weights_path = _resolve_path(
        cap_weights_override if cap_weights_override is not None else capacity_block.get("weights"), base_dir
    )
    cap_weights = (
        _load_snapshot_series(weights_path, snapshots, f"{direction} capacity weights")
        if weights_path is not None
        else None
    )

    if cap_weights is not None and cap_level is None:
        logger.warning(
            "%s capacity weights were provided without a capacity level; the weights will be ignored.", direction
        )

    return DirectionTariff(volumetric=volumetric, cap_level=cap_level, cap_weights=cap_weights)


def load_tariffs(
    config: dict,
    snapshots: pd.DatetimeIndex,
    cli_overrides: dict | None = None,
    base_dir: str | Path | None = None,
) -> Tariffs:
    """Build a Tariffs object from a YAML ``tariffs`` block merged with CLI overrides.

    The expected YAML structure is::

        tariffs:
          withdrawal:
            volumetric: path/to/vol_withdrawal.csv      # EUR/MWh, snapshot-indexed
            capacity:
              level: 5294.0                             # EUR/MW-month
              weights: path/to/weights_withdrawal.csv   # hourly weights, snapshot-indexed
          injection:
            volumetric: path/to/vol_injection.csv
            capacity:
              level: 5294.0
              weights: path/to/weights_injection.csv

    Any field may be omitted; injection defaults to no tariff. CLI overrides take
    precedence over the YAML values when provided.

    Args:
    -----
        config: The loaded workflow config dict (may or may not contain a ``tariffs`` key).
        snapshots: The model snapshots used to align and validate the time series.
        cli_overrides: Optional dict with keys ``vol_withdrawal``, ``vol_injection``,
            ``cap_level_withdrawal``, ``cap_weights_withdrawal``, ``cap_level_injection``,
            ``cap_weights_injection``. Missing/None values fall back to the YAML config.

    Returns:
    --------
        A populated Tariffs object.
    """
    cli_overrides = cli_overrides or {}
    tariffs_block = (config or {}).get("tariffs", {}) or {}

    withdrawal = _build_direction(
        yaml_block=tariffs_block.get("withdrawal", {}),
        snapshots=snapshots,
        vol_override=cli_overrides.get("vol_withdrawal"),
        cap_level_override=cli_overrides.get("cap_level_withdrawal"),
        cap_weights_override=cli_overrides.get("cap_weights_withdrawal"),
        direction="withdrawal",
        base_dir=base_dir,
    )
    injection = _build_direction(
        yaml_block=tariffs_block.get("injection", {}),
        snapshots=snapshots,
        vol_override=cli_overrides.get("vol_injection"),
        cap_level_override=cli_overrides.get("cap_level_injection"),
        cap_weights_override=cli_overrides.get("cap_weights_injection"),
        direction="injection",
        base_dir=base_dir,
    )

    tariffs = Tariffs(withdrawal=withdrawal, injection=injection)

    if tariffs.is_empty:
        logger.info("No grid tariffs configured; running without tariffs.")
    else:
        for direction in DIRECTIONS:
            dt = getattr(tariffs, direction)
            logger.info(
                "%s tariffs -> volumetric: %s, capacity level: %s, weighted: %s",
                direction,
                "yes" if dt.has_volumetric else "no",
                dt.cap_level if dt.has_capacity else "no",
                "yes" if (dt.has_capacity and dt.cap_weights is not None) else "no",
            )

    return tariffs


def add_tariff_cli_arguments(parser) -> None:
    """Add direction-suffixed tariff CLI flags to an argparse parser.

    These flags override the corresponding values in the YAML ``tariffs`` block when provided.
    """
    parser.add_argument(
        "--vol-tou-withdrawal", type=Path, default=None, help="Path to the withdrawal volumetric tariff [€/MWh] CSV."
    )
    parser.add_argument(
        "--vol-tou-injection", type=Path, default=None, help="Path to the injection volumetric tariff [€/MWh] CSV."
    )
    parser.add_argument(
        "--cap-tariff-withdrawal", type=float, default=None, help="Withdrawal capacity tariff level [€/MW-month]."
    )
    parser.add_argument(
        "--cap-weights-withdrawal",
        type=Path,
        default=None,
        help="Path to the snapshot-indexed hourly weights CSV for the withdrawal capacity tariff.",
    )
    parser.add_argument(
        "--cap-tariff-injection", type=float, default=None, help="Injection capacity tariff level [€/MW-month]."
    )
    parser.add_argument(
        "--cap-weights-injection",
        type=Path,
        default=None,
        help="Path to the snapshot-indexed hourly weights CSV for the injection capacity tariff.",
    )


def tariff_cli_overrides(args) -> dict:
    """Collect tariff CLI overrides from parsed argparse args into the load_tariffs override dict."""
    return {
        "vol_withdrawal": args.vol_tou_withdrawal,
        "vol_injection": args.vol_tou_injection,
        "cap_level_withdrawal": args.cap_tariff_withdrawal,
        "cap_weights_withdrawal": args.cap_weights_withdrawal,
        "cap_level_injection": args.cap_tariff_injection,
        "cap_weights_injection": args.cap_weights_injection,
    }
