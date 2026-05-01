import argparse
import logging
from pathlib import Path

import pandas as pd

from demoses_grid_tariffs.electricity_plots import plot_power_flow_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """CLI wrapper for generating plots from power flow results."""
    parser = argparse.ArgumentParser(description="Generate plots from power flow results.")
    parser.add_argument(
        "--network-path",
        type=Path,
        required=True,
        help="Path to the network's Excel file."
    )
    parser.add_argument(
        "--elec-results-dir",
        type=Path,
        required=True,
        help="Directory containing power flow results for the current experiment."
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="Unnamed experiment",
        help="Name of the experiment (used for labeling in comparative histograms)."
    )
    parser.add_argument(
        "--base-results-dir",
        type=Path,
        default=None,
        help="Directory containing base case power flow results used for making relative comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the generated plots. ../figures."
    )
    args = parser.parse_args()

    plot_electricity_results(
        network_path=args.network_path,
        elec_results_dir=args.elec_results_dir,
        experiment_name=args.exp_name,
        base_results_dir=args.base_results_dir,
        output_dir=args.output_dir
    )


def plot_electricity_results(
    network_path: Path,
    elec_results_dir: Path,
    experiment_name: str,
    base_results_dir: Path = None,
    output_dir: Path = None,
):
    """Generate plots from electricity model power flow results."""
    if output_dir is None:
        output_dir = elec_results_dir / "figures"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Read power flow results from the experiment directory
    exp_dir = elec_results_dir
    exp_bus_res = pd.read_csv(exp_dir / "res_bus/vm_pu_with_names.csv", index_col=0, parse_dates=True)
    exp_line_res = pd.read_csv(exp_dir / "res_line/loading_percent_with_names.csv", index_col=0, parse_dates=True)
    exp_trafo_res = pd.read_csv(exp_dir / "res_trafo/loading_percent_with_names.csv", index_col=0, parse_dates=True)

    # Load base case results if the path is provided
    b_bus_res, b_line_res, b_trafo_res = None, None, None
    if base_results_dir:
        logger.info(f"Loading base case results from: {base_results_dir}")
        base_dir = base_results_dir
        b_bus_res = pd.read_csv(base_dir / "res_bus/vm_pu_with_names.csv", index_col=0, parse_dates=True)
        b_line_res = pd.read_csv(base_dir / "res_line/loading_percent_with_names.csv", index_col=0, parse_dates=True)
        b_trafo_res = pd.read_csv(base_dir / "res_trafo/loading_percent_with_names.csv", index_col=0, parse_dates=True)

    # Plot the results.
    logger.info("Generating plots from power flow results...")
    logger.info(f"Saving plots to: {figures_dir}")

    plot_power_flow_results(
        bus_result=exp_bus_res,
        line_result=exp_line_res,
        trafo_result=exp_trafo_res,
        experiment_name=experiment_name,
        network_path=network_path,
        output=figures_dir,
        base_bus_result=b_bus_res,
        base_line_result=b_line_res,
        base_trafo_result=b_trafo_res,
    )

    logger.info(" ============ Successfully generated power flow results plots 🎉🎉🎉 ========== ")


if __name__ == "__main__":
    main()
