import argparse
import datetime as dt
import logging
import shutil
from pathlib import Path

import yaml

import pandas as pd
from pyprojroot import here

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Stedin 2026 volumetric (kWh) transport tariffs, converted to EUR/MWh (x1000).
# Source: Stedin "Tarieven 2026 Elektriciteit Grootverbruik", Table 3 (transportvergoeding, p.6).
# 'normaal' = weekdays 07:00-23:00; 'laag' = all other hours and Dutch public holidays.
# NOTE: at TS / Trafo HS+TS/MS (>1500 kW, i.e. most >5 MW DHN assets) there is NO volumetric
# transport tariff in 2026, so a volumetric tariff there (scenarios 4 and 6) is hypothetical and
# uses the MS rate as a stand-in.
STEDIN_2026_VOLUMETRIC_EUR_PER_MWH = {
    "MS": 19.8,          # 0.0198 EUR/kWh, flat (normaal == laag), 151-1500 kW (and Trafo MS/LS)
    "LS_normaal": 74.9,  # 0.0749 EUR/kWh, weekdays 07:00-23:00, <= 50 kW
    "LS_laag": 46.0,     # 0.0460 EUR/kWh, other hours and holidays, <= 50 kW
}


def main() -> None:
    """Generate a scenario's grid-tariff parameter files into its results inputs folder.

    Given a scenario config and a withdrawal weighting schedule, this writes all tariff parameter
    files into ``results/<config-stem>/inputs/tariff_params/``:
      - weighting_factors_withdrawal.csv  (copy of the input schedule, for provenance)
      - weighting_factors_injection.csv   (reversed/rank-mirrored injection schedule)
      - weights_withdrawal.csv, weights_injection.csv   (8760-hour capacity weight profiles)
      - vol_flat.csv, vol_tou_withdrawal.csv, vol_tou_injection.csv   (volumetric profiles, EUR/MWh)
      - figures/weighting_heatmap.png

    The run year and number of snapshots are read from the config (scenario_params.year and
    model_params.num_snapshots) unless overridden. Escalate the 2026 base levels to the run year
    with --annual-growth-rate (see escalation_factor for FIEN26 values).

    Example:
        python src/demoses_grid_tariffs/generate_tariffs.py \
            --config configs/scenarios/scenario_5_tou_capacity_both.yaml \
            --withdrawal-schedule data/weighting_factors_distribution.csv \
            --base-rate 19.8 --annual-growth-rate 0.0
    """
    parser = argparse.ArgumentParser(description="Generate a scenario's grid-tariff parameter files.")
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Scenario config YAML; its stem names the output folder and provides year/num_snapshots.",
    )
    parser.add_argument(
        "--withdrawal-schedule", type=Path, required=True,
        help="Withdrawal weighting schedule CSV, e.g. data/weighting_factors_distribution.csv.",
    )
    parser.add_argument(
        "--base-rate", type=float, default=STEDIN_2026_VOLUMETRIC_EUR_PER_MWH["MS"],
        help="Base volumetric rate EUR/MWh (default: Stedin 2026 MS = 19.8).",
    )
    parser.add_argument("--base-year", type=int, default=2026, help="Year the base levels refer to (default 2026).")
    parser.add_argument(
        "--annual-growth-rate", type=float, default=0.0,
        help="Real CAGR to escalate base_year -> run year (FIEN26 ~3.7-6.3 pct/yr; see escalation_factor).",
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Override run year (default: config scenario_params.year)."
    )
    parser.add_argument(
        "--num-snapshots", type=int, default=None,
        help="Override number of snapshots (default: config model_params.num_snapshots).",
    )
    parser.add_argument("--no-holidays", action="store_true", help="Do not treat Dutch public holidays as weekend.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    year = args.year if args.year is not None else int(config["scenario_params"]["year"])
    num_snapshots = (
        args.num_snapshots if args.num_snapshots is not None else int(config["model_params"]["num_snapshots"])
    )
    include_holidays = not args.no_holidays

    out_dir = here() / "results" / args.config.stem / "inputs" / "tariff_params"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=num_snapshots, freq="h")

    # Schedules: keep a copy of the withdrawal schedule for provenance and derive the injection one.
    withdrawal_schedule = out_dir / "weighting_factors_withdrawal.csv"
    injection_schedule = out_dir / "weighting_factors_injection.csv"
    shutil.copyfile(args.withdrawal_schedule, withdrawal_schedule)
    reverse_weight_schedule(args.withdrawal_schedule, injection_schedule)

    # Capacity weight profiles (withdrawal + injection).
    write_weight_profile(withdrawal_schedule, year, num_snapshots, out_dir / "weights_withdrawal.csv", include_holidays)
    write_weight_profile(injection_schedule, year, num_snapshots, out_dir / "weights_injection.csv", include_holidays)

    # Volumetric profiles (flat base case + ToU withdrawal + ToU injection).
    generate_flat_vol_tariff(
        snapshots, args.base_rate, args.base_year, year, args.annual_growth_rate
    ).rename_axis("snapshots").to_csv(out_dir / "vol_flat.csv")
    generate_tou_vol_tariffs(
        snapshots, withdrawal_schedule, args.base_rate, args.base_year, year, args.annual_growth_rate, include_holidays
    ).rename_axis("snapshots").to_csv(out_dir / "vol_tou_withdrawal.csv")
    generate_tou_vol_tariffs(
        snapshots, injection_schedule, args.base_rate, args.base_year, year, args.annual_growth_rate, include_holidays
    ).rename_axis("snapshots").to_csv(out_dir / "vol_tou_injection.csv")

    # Comparison heatmap (withdrawal vs reversed injection).
    plot_weight_schedules_heatmap(
        {"Withdrawal": withdrawal_schedule, "Injection (reversed)": injection_schedule},
        fig_dir / "weighting_heatmap.png",
        title=f"{args.config.stem}: capacity-tariff weighting (withdrawal vs injection)",
    )
    logger.info(
        f"Tariff params for '{args.config.stem}' written to {out_dir} (year {year}, growth {args.annual_growth_rate})."
    )


def escalation_factor(base_year: int, target_year: int, annual_growth_rate: float) -> float:
    """Compound escalation factor (1 + g)^(target_year - base_year) for projecting tariff levels.

    Tariff inputs in this project default to Stedin's 2026 levels. To project them to a future year,
    multiply by this factor with a real compound annual growth rate g.

    Applying FIEN26: the periodic system-operator report FIEN26 (PwC Strategy&, 2026; tariff chapter,
    p.30) projects real electricity *network* tariffs to grow about 3.7-6.3 percent per year between
    2026 and 2040 (CAGR varies by user group), i.e. roughly 1.7-2.3x by 2040. Extrapolating the same
    CAGR to 2050 (24 years from the 2026 base) gives:
        g = 0.037 -> factor ~= 2.4x   (FIEN26 low)
        g = 0.047 -> factor ~= 3.0x   (FIEN26 mid)
        g = 0.063 -> factor ~= 4.4x   (FIEN26 high)
    There is no official 2050 figure, so treat the chosen g as an explicit, citable assumption and,
    for consistency, apply the SAME factor to the capacity tariff level (EUR/MW-month) in the scenario
    configs. Example: escalation_factor(2026, 2050, 0.047).
    """
    return (1.0 + annual_growth_rate) ** (target_year - base_year)


def generate_tou_vol_tariffs(
    snapshots: pd.DatetimeIndex,
    weight_schedule_csv: Path,
    base_rate_eur_per_mwh: float = STEDIN_2026_VOLUMETRIC_EUR_PER_MWH["MS"],
    base_year: int = 2026,
    target_year: int = 2050,
    annual_growth_rate: float = 0.0,
    include_holidays: bool = True,
) -> pd.Series:
    """Build a time-of-use volumetric network tariff profile in EUR/MWh.

    Implements the regulator's weighted-energy charge (kWh_gewogen): the per-MWh price in each hour
    is the base rate times that hour's weighting factor, so energy is priced higher at high-stress
    hours. The base rate is escalated from base_year to target_year by a compound annual growth rate:

        price[t] = base_rate * (1 + annual_growth_rate)^(target_year - base_year) * weight[t]

    By default annual_growth_rate is 0.0, so the profile is at the base-year (2026) level. To project
    to 2050, pass a FIEN26-based growth rate, e.g. annual_growth_rate=0.047 (~3.0x by 2050); see
    escalation_factor() for the FIEN26 low/mid/high values and apply the same factor to the capacity
    tariff level for consistency.

    Args:
    -----
        snapshots: Hourly snapshots to generate the profile for.
        weight_schedule_csv: Month x hour weighting schedule (columns: daytype, month, h1..h24).
            Use the withdrawal (demand-based) schedule for a withdrawal tariff, or the reversed
            schedule for an injection tariff.
        base_rate_eur_per_mwh: Base volumetric rate in EUR/MWh (default: Stedin 2026 MS kWh tariff).
        base_year: Year the base rate refers to (default 2026, the Stedin tariff year).
        target_year: Year to escalate the rate to (default 2050).
        annual_growth_rate: Real compound annual growth rate (e.g. 0.047). Default 0.0 keeps the
            base-year level. See FIEN26 for plausible values (~3.7-6.3 percent per year).
        include_holidays: If True, Dutch public holidays use the weekend rows of the schedule.

    Returns:
    --------
        A snapshot-indexed pandas Series named 'vol_tou_tariffs' in EUR/MWh.
    """
    weights = build_weight_series(
        snapshots,
        load_weight_schedule(weight_schedule_csv),
        include_holidays=include_holidays,
        year=snapshots[0].year,
    )
    factor = escalation_factor(base_year, target_year, annual_growth_rate)
    series = (base_rate_eur_per_mwh * factor) * weights
    series.name = "vol_tou_tariffs"
    return series


def generate_flat_vol_tariff(
    snapshots: pd.DatetimeIndex,
    base_rate_eur_per_mwh: float = STEDIN_2026_VOLUMETRIC_EUR_PER_MWH["MS"],
    base_year: int = 2026,
    target_year: int = 2050,
    annual_growth_rate: float = 0.0,
) -> pd.Series:
    """Build a flat (time-invariant) volumetric network tariff profile in EUR/MWh, escalated.

    Used for the base-case scenario. Defaults to the Stedin 2026 MS kWh tariff (19.8 EUR/MWh).
    """
    factor = escalation_factor(base_year, target_year, annual_growth_rate)
    return pd.Series(base_rate_eur_per_mwh * factor, index=snapshots, name="vol_tou_tariffs")


def easter_sunday(year: int) -> dt.date:
    """Return the date of Easter Sunday for a given year (anonymous Gregorian algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def dutch_public_holidays(year: int) -> set:
    """Return the set of Dutch public holidays treated as non-working days for grid tariffs.

    Matches the holidays listed in the ACM transport-tariff code: Nieuwjaarsdag, 2e Paasdag,
    Koningsdag, Hemelvaartsdag, 2e Pinksterdag, and 1e/2e Kerstdag.
    """
    easter = easter_sunday(year)
    holidays = {
        dt.date(year, 1, 1),                       # Nieuwjaarsdag
        easter + dt.timedelta(days=1),             # 2e Paasdag (Easter Monday)
        easter + dt.timedelta(days=39),            # Hemelvaartsdag (Ascension)
        easter + dt.timedelta(days=50),            # 2e Pinksterdag (Whit Monday)
        dt.date(year, 12, 25),                     # 1e Kerstdag
        dt.date(year, 12, 26),                     # 2e Kerstdag
    }
    kings_day = dt.date(year, 4, 27)               # Koningsdag (26 April if the 27th is a Sunday)
    if kings_day.weekday() == 6:
        kings_day = dt.date(year, 4, 26)
    holidays.add(kings_day)
    return holidays


def load_weight_schedule(schedule_csv: Path) -> dict:
    """Load a month x hour weighting schedule into a dict keyed by (daytype, month).

    The CSV must have columns: daytype ('weekday' or 'weekend'), month (1-12), and h1..h24.
    """
    df = pd.read_csv(schedule_csv, comment="#")
    schedule = {}
    for _, row in df.iterrows():
        key = (str(row["daytype"]).strip().lower(), int(row["month"]))
        schedule[key] = [float(row[f"h{hour}"]) for hour in range(1, 25)]
    return schedule


def build_weight_series(
    snapshots: pd.DatetimeIndex, schedule: dict, include_holidays: bool = True, year: int | None = None
) -> pd.Series:
    """Expand a (daytype, month) x hour weighting schedule into a snapshot-indexed series.

    Weekends (Saturday/Sunday) and, when ``include_holidays`` is True, Dutch public holidays
    use the 'weekend' rows; all other days use the 'weekday' rows. The hour is taken from each
    snapshot (hour 0 maps to schedule column h1, hour 23 to h24).
    """
    year = year or snapshots[0].year
    holidays = dutch_public_holidays(year) if include_holidays else set()
    values = []
    for timestamp in snapshots:
        is_non_working = timestamp.weekday() >= 5 or (timestamp.date() in holidays)
        daytype = "weekend" if is_non_working else "weekday"
        values.append(schedule[(daytype, timestamp.month)][timestamp.hour])
    return pd.Series(values, index=snapshots, name="value")


def write_weight_profile(
    schedule_csv: Path,
    year: int,
    num_snapshots: int,
    output_path: Path,
    include_holidays: bool = True,
) -> pd.Series:
    """Build a snapshot-indexed capacity-tariff weight profile from a schedule CSV and save it.

    The output is a CSV with a 'snapshots' datetime index and a single 'value' column, ready to
    be passed to the model as ``--cap-weights-withdrawal`` / ``--cap-weights-injection``.
    """
    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=num_snapshots, freq="h")
    schedule = load_weight_schedule(schedule_csv)
    series = build_weight_series(snapshots, schedule, include_holidays=include_holidays, year=year)
    series.index.name = "snapshots"
    series.to_csv(output_path)
    logger.info(f"Saved weight profile ({len(series)} snapshots) to {output_path}")
    return series


def reverse_weight_schedule(schedule_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Build an injection weight schedule by rank-reversing a (withdrawal) schedule.

    Each weight is mapped to its mirror within the sorted set of distinct weight levels, so the
    highest-demand hours (highest withdrawal weight) become the lowest injection weight and vice
    versa. This makes the injection charge high when local demand is low (when reverse-flow /
    injection congestion typically occurs), instead of rewarding injection at those hours.

    NOTE: This rank-reversal is a transparent first-order proxy. A fully cost-reflective injection
    weight would be derived from the actual net-injection (reverse-flow) duration at the relevant
    network level; CHP export that coincides with the demand peak is grid-relieving and is not
    captured by a mechanical reversal.
    """
    df = pd.read_csv(schedule_csv, comment="#")
    hour_cols = [f"h{hour}" for hour in range(1, 25)]
    levels = sorted(pd.unique(df[hour_cols].to_numpy().ravel()))
    mirror = {value: levels[len(levels) - 1 - i] for i, value in enumerate(levels)}
    df[hour_cols] = df[hour_cols].apply(lambda col: col.map(mirror))
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved reversed (injection) weight schedule to {output_csv}")
    return df


def plot_weight_schedules_heatmap(
    schedule_csvs: dict,
    output_path: Path,
    cmap: str = "YlOrRd",
    annotate: bool = True,
    title: str = "Capacity-tariff weighting factors (kW_maxgewogen)",
) -> None:
    """Plot month x hour weighting-factor heatmaps in the matrix form of the Tarievencode table.

    Renders one column per schedule (e.g. withdrawal vs injection) and one row per day type
    (weekday, weekend), so the reverse relationship between withdrawal and injection weights is
    visible at a glance. Cell color encodes the weighting factor (0-1); values are annotated.

    Args:
    -----
        schedule_csvs: Mapping of column label -> schedule CSV path (columns: daytype, month, h1..h24).
        output_path: Path to save the figure (e.g. a .png).
        cmap: Matplotlib colormap (sequential; higher weight = warmer).
        annotate: If True, write each weight value in its cell (as in the source table).
        title: Figure super-title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    hour_cols = [f"h{hour}" for hour in range(1, 25)]
    daytypes = ["weekday", "weekend"]
    labels = list(schedule_csvs)

    fig, axes = plt.subplots(
        len(daytypes), len(labels), figsize=(5.2 * len(labels), 4.4 * len(daytypes)), squeeze=False
    )
    image = None
    for row, daytype in enumerate(daytypes):
        for col, label in enumerate(labels):
            df = pd.read_csv(schedule_csvs[label], comment="#")
            sub = df[df["daytype"].str.lower() == daytype].set_index("month").sort_index()[hour_cols]
            matrix = sub.to_numpy(dtype=float)
            ax = axes[row][col]
            image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
            ax.set_title(f"{label} - {daytype}", fontsize=10)
            ax.set_xticks(range(24))
            ax.set_xticklabels(range(1, 25), fontsize=6)
            ax.set_yticks(range(len(sub.index)))
            ax.set_yticklabels([month_labels[m - 1] for m in sub.index], fontsize=7)
            ax.set_xlabel("Hour of day", fontsize=8)
            if col == 0:
                ax.set_ylabel("Month", fontsize=8)
            if annotate:
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=4.0, color="black")

    fig.colorbar(image, ax=axes, shrink=0.6, label="weighting factor")
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved weighting-factor heatmap to {output_path}")


if __name__ == "__main__":
    main()
