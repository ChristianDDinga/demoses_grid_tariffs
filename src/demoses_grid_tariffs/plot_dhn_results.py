import argparse
import logging
from pathlib import Path

import pypsa

from demoses_grid_tariffs.dhn_plots import plot_dhn_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Generate plots from the optimized district heating network model."""
    parser = argparse.ArgumentParser(description="Generate plots from the optimized district heating network model.")
    parser.add_argument(
        "--heat-results-dir",
        type=Path,
        required=True,
        help="Path to heat results directory (../<scenario_name>/02_heat_outputs/least_cost/ or .../spores/spore_1/)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the heat model plots. If not specified, defaults to <heat_results_dir>/figures.",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.heat_results_dir / "figures"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the solved/optimized network.
    logger.info(f"Reading heat model results from: {args.heat_results_dir}")
    solved_network_path = args.heat_results_dir / "solved_pypsa_network.nc"
    if not solved_network_path.exists():
        raise FileNotFoundError(f"Solved network not found at: {solved_network_path}")

    n = pypsa.Network()
    n.import_from_netcdf(solved_network_path)

    # Plot results.
    logger.info("Generating heat model plots...")
    logger.info(f"Generated plots will be saved to: {args.output_dir}")
    plot_dhn_results(n, output=args.output_dir)

    logger.info(" =============== Successfully generated heat model plots 🎉🎉🎉 ============== ")


if __name__ == "__main__":
    main()
