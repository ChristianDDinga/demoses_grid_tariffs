import argparse
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from demoses_grid_tariffs.data_processing import (
    prepare_and_save_heat_model_csv_data,
    prepare_network_component_files,
)
from demoses_grid_tariffs.helper_functions import calculate_heatpump_cop, customize_and_save_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_RESULTS_DIR = Path("results")


def main()-> None:
    """Orchestrates the preparation of a self-contained scenario directory for the heat model."""
    parser = argparse.ArgumentParser(description="Prepare a directory will all data needed to run the heat model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the workflow_config.yaml file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path where heat inputs will be saved (e.g, ../<scenario_name>/01_heat_inputs/)",
    )
    parser.add_argument("--overwrite", action="store_true", help="If set, overwrites the output directory if exists.")
    args = parser.parse_args()

    # Setup the output directory
    if args.output_dir.exists():
        if args.overwrite:
            logger.warning(f"Output directory {args.output_dir} exists and --overwrite is set. Overwriting ...")
            shutil.rmtree(args.output_dir)  # Delete and recreate a fresh directory.
        else:
            raise FileExistsError(
                f"{args.output_dir=} already exists. Use --overwrite in the main orchestrator to replace it.",
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing heat inputs in: {args.output_dir}")

    # Load Configuration
    with open(args.config, "r") as f:
        config_file = yaml.safe_load(f)
    scenario_params = config_file["scenario_params"]
    data = config_file["data_sources"]
    num_snapshots = config_file["model_params"]["num_snapshots"]
    adjustments = config_file.get("scenario_adjustments", {})

    # Standardize timeseries indices and slice to the correct number of snapshots
    year = scenario_params["year"]
    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=num_snapshots, freq="h")

    # Prepare csv data files
    processed_dfs = prepare_and_save_heat_model_csv_data(data, scenario_params, adjustments, snapshots, args.output_dir)

    # Prepare network component CSV files
    prepare_network_component_files(data, scenario_params, adjustments, args.output_dir)

    # Generate heat input data plots for quick diagnostics
    figure_folder = args.output_dir / "figures"
    figure_folder.mkdir(exist_ok=True)
    plot_heat_input_data(processed_dfs, figure_folder)

    # Copy configuration files for reproducibility
    copy_all_configs(
        main_config_path=args.config,
        main_config_data=config_file,
        run_name=args.config.stem,
        base_results_dir=DEFAULT_RESULTS_DIR,
    )

    logger.info(" ============ Successfully prepared and plotted heat inputs 🎉🎉🎉 ============ ")


def copy_all_configs(main_config_path: Path, main_config_data: dict, run_name: str, base_results_dir: Path) -> None:
    """Copies all relevant configuration files for reproducibility."""
    destination_dir = base_results_dir / run_name / "00_configs"

    logger.info(f"Copying configuration files to {destination_dir} for reproducibility...")
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Copy the main workflow config file
    try:
        shutil.copy(main_config_path, destination_dir)
        logger.info(f"  - Copied main config: {main_config_path.name}")
    except Exception as e:
        logger.error(f"Could not copy main config file {main_config_path}. Error: {e}")

    # Check for and copy the SPORES config file(s), if applicable
    model_params = main_config_data.get("model_params", {})
    is_spores_mode = model_params.get("heat_model_mode") == "spores"

    if is_spores_mode:
        spores_config_str = model_params.get("spores_config")
        if not spores_config_str:
            logger.warning("Workflow is in 'spores' mode, but 'spores_config' path is not defined.")
            return

        spores_config_path = Path(spores_config_str)

        if spores_config_path.exists():
            if spores_config_path.is_file():
                shutil.copy(spores_config_path, destination_dir)
                logger.info(f"  - Copied SPORES config file: {spores_config_path.name}")
            elif spores_config_path.is_dir():
                dest_subdir = destination_dir / spores_config_path.name
                if dest_subdir.exists():
                    shutil.rmtree(dest_subdir)
                shutil.copytree(spores_config_path, dest_subdir)
                logger.info(f"  - Copied SPORES config directory: {spores_config_path.name}")
        else:
            logger.error(f"SPORES config path specified but not found: {spores_config_path}")


def plot_heat_input_data(
    dataframes_to_plot: dict, figure_folder: Path, figsize: tuple =(15, 8), dpi: int=300, extension: str  ="png",
) -> None:
    """Plot some input data for the heat model from prepared sources.

    Args:
    -----
        dataframes_to_plot: A dictionary of dataframes to plot.
        figure_folder: The output folder to save the figures.
        figsize: The figure size for the plots.
        dpi: The resolution of the saved figures.
        extension: The file extension for saving the figures (e.g., 'png', 'pdf').
    """
    snapshots = dataframes_to_plot["demand"].index

    # Plot heat demand
    demand_to_plot = dataframes_to_plot["demand"].copy()
    demand_to_plot["Total"] = demand_to_plot.sum(axis=1)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Plot all columns, then customize the 'Total' line color
    demand_to_plot.plot(ax=ax, legend=False)
    ax.get_lines()[-1].set_color("grey") # Make the last line (Total) grey
    ax.get_lines()[-1].set_linewidth(2)
    customize_and_save_plot(
        ax=ax,
        output_dir=figure_folder,
        filename=f"heat_demand.{extension}",
        ylabel="Heat demand [MW]",
        ylim=(0, demand_to_plot["Total"].max() * 1.05),
        legend_handles=ax.get_lines(),  # We handle legend manually here to customize color
        legend_labels=demand_to_plot.columns.tolist(),
    )

    # Plot electricity and hydrogen prices
    for price_ts, col in [("electricity_price", "blue"), ("hydrogen_price", "orange")]:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        price_data = dataframes_to_plot[price_ts]
        price_data.plot(ax=ax, color=col, legend=False) # Use pandas plot
        customize_and_save_plot(
            ax=ax,
            output_dir=figure_folder,
            filename=f"{price_ts}.{extension}",
            ylabel=f"{price_ts.replace('_', ' ').capitalize()} [EUR/MWh]",
            add_legend=False,
        )

    # Plot temperature
    temperature = dataframes_to_plot["temperature"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    temperature.plot(ax=ax, color={"ambient": "purple", "ground": "brown"}, legend=False)
    customize_and_save_plot(
        ax=ax,
        output_dir=figure_folder,
        filename=f"temperature.{extension}",
        ylabel="Temperature [°C]",
        add_legend=False,
        ylim=(temperature.min().min() * 1.05, temperature.max().max() * 1.1),
    )

    # Plot heat pump COP
    # First, assemble the calculated COPs into a DataFrame
    temperature_data = dataframes_to_plot["temperature"]
    cop_df = pd.DataFrame(index=snapshots)
    cop_df["ASHP"] = calculate_heatpump_cop(tech_carrier="ashp", temp_source=temperature_data["ambient"].values)
    if "ground" in temperature_data.columns:
        cop_df["GSHP"] = calculate_heatpump_cop(tech_carrier="gshp", temp_source=temperature_data["ground"].values)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cop_df.plot(ax=ax, color={"ASHP": "purple", "GSHP": "brown"}, legend=False) # Use pandas plot
    customize_and_save_plot(
        ax=ax,
        output_dir=figure_folder,
        filename=f"heatpump_cop.{extension}",
        ylabel="Heat pump COP [-]",
        y_axis_formatter="%.1f",
        add_legend=False,
    )

    # Plot solar availability
    solar_availability = dataframes_to_plot["solar_availability"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    solar_availability.plot(ax=ax, color="gold", legend=False) # Use pandas plot
    customize_and_save_plot(
        ax=ax,
        output_dir=figure_folder,
        filename=f"solar_thermal.{extension}",
        ylabel="Solar thermal capacity factor [-]",
        ylim=(0, 1.05),
        y_axis_formatter="%.1f",
        add_legend=False,
    )

    # Plot static prices
    # First, assemble the static prices into a time-series DataFrame
    static_prices = dataframes_to_plot["static_prices"]
    static_prices_ts = pd.DataFrame(index=snapshots)
    for carrier in static_prices.columns:
        static_prices_ts[carrier] = static_prices[carrier].iloc[0]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    static_prices_ts.plot(ax=ax, legend=False) # Use pandas plot
    customize_and_save_plot(
        ax=ax,
        output_dir=figure_folder,
        filename=f"static_prices.{extension}",
        ylabel="Resource price [EUR/MWh]",
        legend_title="Carrier",
    )

    logger.info(f"Generated heat input data plots in {figure_folder}")


if __name__ == "__main__":
    main()
