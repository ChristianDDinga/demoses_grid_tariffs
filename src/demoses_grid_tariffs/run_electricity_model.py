import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from demoses_grid_tariffs.electricity_model import run_power_flow
from demoses_grid_tariffs.helper_functions import generate_powerflow_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HIGH_VOLTAGE_SUBSTATIONS = ["Zuidwijk_150_load", "Zuidwijk_150_sgen", "Voorburg_150_load", "Voorburg_150_sgen"]


def main():
    """CLI wrapper for running the electricity model."""
    parser = argparse.ArgumentParser(description="Run the electricity distribution network power flow model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the main workflow_config.yaml file.")
    parser.add_argument(
        "--network-path",
        type=Path,
        required=True,
        help="Path to the excel file defining the pandapower electricity network model.",
    )
    parser.add_argument(
        "--profiles-path",
        type=Path,
        required=True,
        help="Path to the CSV file containing the power profiles for sgen and load components in the network.",
    )
    parser.add_argument(
        "--dc-power-flow",
        action="store_true",
        help="If set, uses DC power flow instead of AC power flow for the simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path where the power flow simulation results will be saved.",
    )
    parser.add_argument(
        "--exclude-hv-power",
        action="store_true",
        help="If set, excludes the power profiles for high voltage substations in the power flow simulation.",
    )
    args = parser.parse_args()

    run_electricity_model(
        config_path=args.config,
        network_path=args.network_path,
        profiles_path=args.profiles_path,
        output_dir=args.output_dir,
        exclude_hv_power=args.exclude_hv_power,
        dc_power_flow=args.dc_power_flow
    )


def run_electricity_model(
    config_path: Path,
    network_path: Path,
    profiles_path: Path,
    output_dir: Path,
    exclude_hv_power: bool = False,
    dc_power_flow: bool = False
):
    """Run the electricity distribution network power flow model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all required input data
    logger.info(f"Loading all required input data from: {network_path} and {profiles_path}")
    power_profiles = pd.read_csv(profiles_path, index_col=0)

    # Multiply the power profiles for high voltage substations by 0 if exclude_hv_power flag is set.
    if exclude_hv_power:
        logger.info(
            f"The --exclude-hv-power flag is set. Setting power at {HIGH_VOLTAGE_SUBSTATIONS=} to 0."
        )
        for hv_substation in HIGH_VOLTAGE_SUBSTATIONS:
            power_profiles[hv_substation] *= 0
    else:
        logger.info(
            f"The --exclude-hv-power flag is not set. Including {HIGH_VOLTAGE_SUBSTATIONS=} in the simulation."
        )

    # Load the config file to get the year for creating a datetime index.
    with open(config_path, "r") as f:
        year = yaml.safe_load(f)["scenario_params"]["year"]

    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=len(power_profiles), freq="h")

    # Run the power flow simulation.
    logger.info("Running power flow simulation...")
    run_power_flow(network_path, power_profiles, snapshots, dc_power_flow, output_dir)
    logger.info(" ============= Successfully completed power flow simulation 🎉🎉🎉 ============ ")

    # Generate electricity network statistics.
    logger.info("Generating electricity network statistics...")
    generate_powerflow_statistics(output_dir)
    logger.info(f"📉📉📉Successfully generated electricity network statistics file at: {output_dir}📉📉📉")


if __name__ == "__main__":
    main()
