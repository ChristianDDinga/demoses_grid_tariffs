import argparse
import logging
from pathlib import Path

import pandas as pd
import pypsa
import yaml

from demoses_grid_tariffs.dhn_model import build_district_heating_network, optimize_district_heating_network
from demoses_grid_tariffs.helper_functions import (
    get_electricity_consumption_of_assets,
    get_electricity_generation_of_assets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Builds, runs, and writes out the least-cost district heating optimization model."""
    parser = argparse.ArgumentParser(description="Build and run the least-cost district heating model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the workflow_config.yaml file.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing heat model inputs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save heat results.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Build and solve the least-cost model.
    solved_lc_network = build_and_solve_least_cost_network(args.input_dir, config)

    # 2. Save the results to the 'least_cost' subdirectory.
    save_network_results(solved_lc_network, args.output_dir / "least_cost")

    logger.info(" ============ Successfully completed least-cost heat model run 🎉🎉🎉 ============ ")


def build_and_solve_least_cost_network(input_dir: Path, config: dict) -> pypsa.Network:
    """Builds the district heating network model, solves for least-cost solution, and returns the solved network."""
    year = config["scenario_params"]["year"]
    num_snapshots = config["model_params"]["num_snapshots"]
    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=num_snapshots, freq="h")

    logger.info(f"Building the district heating network model from inputs in {input_dir}...")
    network, model = build_district_heating_network(
        csv_folder=input_dir / "network",
        temperature=pd.read_csv(input_dir / "temperature.csv", index_col="snapshots", parse_dates=True),
        heat_demand=pd.read_csv(input_dir / "demand.csv", index_col="snapshots", parse_dates=True),
        hydrogen_price=pd.read_csv(input_dir / "hydrogen_price.csv", index_col="snapshots", parse_dates=True),
        electricity_price=pd.read_csv(input_dir / "electricity_price.csv", index_col="snapshots", parse_dates=True),
        solar_availability=pd.read_csv(input_dir / "solar_availability.csv", index_col="snapshots", parse_dates=True),
        static_prices=pd.read_csv(input_dir / "static_prices.csv", index_col="year"),
        snapshots=snapshots,
    )
    logger.info("Successfully built the district heating network optimization model.")

    logger.info("Running least-cost optimization of the heat model...")
    solver_options = config["model_params"]["solver_options"]
    least_cost_network = optimize_district_heating_network(network, model, solver_options)
    logger.info("Least-cost optimization complete.")

    return least_cost_network


def save_network_results(network: pypsa.Network, output_dir: Path) -> None:
    """A helper function to save the key outputs from a solved network."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving optimized heat network model results (including the pypsa network .nc file) to {output_dir}")
    network.statistics().to_csv(output_dir / "statistics.csv")
    network.export_to_netcdf(output_dir / "solved_pypsa_network.nc")
    get_electricity_generation_of_assets(network).to_csv(output_dir / "electricity_generation.csv")
    get_electricity_consumption_of_assets(network).to_csv(output_dir / "electricity_consumption.csv")
    logger.info(f"Finished saving optimized heat network model results to {output_dir}")


if __name__ == "__main__":
    main()
