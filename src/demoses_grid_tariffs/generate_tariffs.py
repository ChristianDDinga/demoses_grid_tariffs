import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Generates the volumetric TOU tariff profile and saves it to a CSV."""
    parser = argparse.ArgumentParser(description="Generate volumetric TOU tariffs based on demand profiles.")
    parser.add_argument("--demand-csv", type=Path, required=True, help="Path to the (electrified) demand.csv file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to save the vol_tou_tariffs.csv file.")
    args = parser.parse_args()

    demand_df = pd.read_csv(args.demand_csv, index_col="snapshots", parse_dates=True)
    vol_tou_tariffs = generate_tou_tariffs(demand_df)
    vol_tou_tariffs.to_csv(args.output_dir / "vol_tou_tariffs.csv")

    logger.info(f"Successfully generated and saved volumetric TOU tariffs file in {args.output_dir} 🎉🎉🎉")


def generate_tou_tariffs(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a TOU tariff profile based on demand intensity and seasonality.
    
    It categorizes each hour into 'off', 'mid', or 'on' peak based on demand,
    and applies different prices for each taking into account winter vs.
    non-winter periods.
    """
    df = demand_df.copy()

    # Total
    df["total"] = df.sum(axis=1)
    
    # 1. Define Seasons (Winter: Nov, Dec, Jan, Feb, Mar)
    # Adjust the month list if your region's winter definition differs
    df['is_winter'] = df.index.month.isin([11, 12, 1, 2, 3])

    # 2. Define Peak Categories based on Demand Intensity
    # We use quantiles so the "Highest Demand" hours always get the "On-Peak" price
    # Top 20% = On-Peak, Middle 40% = Mid-Peak, Bottom 40% = Off-Peak
    low_threshold = df["total"].quantile(0.40)
    high_threshold = df["total"].quantile(0.80)

    def get_peak_type(val):
        if val <= low_threshold:
            return 'off'
        elif val <= high_threshold:
            return 'mid'
        else:
            return 'on'

    df['peak_type'] = df["total"].apply(get_peak_type)

    # 3. Define Price in €/MW
    # Prices: {Season: {Peak: Price}}
    prices = {
        True: { # Winter
            'off': 4.91,
            'mid': 8.52,
            'on': 12.07,
        },
        False: { # Non-Winter
            'off': 4.97,
            'mid': 9.06,
            'on': 12.68
        }
    }

    # 4. Apply the pricing logic
    df['vol_tou_tariff'] = df.apply(
        lambda row: prices[row['is_winter']][row['peak_type']], axis=1
    )

    return df[['vol_tou_tariff']]


if __name__ == "__main__":
    main()
