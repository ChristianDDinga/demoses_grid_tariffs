import argparse
import logging
import shutil
import pandas as pd
import pyomo.environ as pyo
import yaml

from pathlib import Path
from cronian.results import extract_prosumer_dispatch

from demoses_grid_tariffs.data_processing import (
    prepare_and_save_heat_model_csv_data,
    prepare_network_component_files,
)

from demoses_grid_tariffs.dhn_plots import plot_dhn_results
from demoses_grid_tariffs.helper_functions import fill_path_wildcards, plot_capacity_tariff, plot_vol_tou_tariffs
from demoses_grid_tariffs.plot_electricity_results import plot_electricity_results
from demoses_grid_tariffs.prepare_dhn_input import plot_heat_input_data
from demoses_grid_tariffs.prepare_electricity_input import (
    add_dhn_assets_to_pandapower_excel_sheets,
    split_bidirectional_profiles,
)
from demoses_grid_tariffs.run_dhn_model import build_and_solve_least_cost_network, save_network_results
from demoses_grid_tariffs.run_electricity_model import run_electricity_model
from demoses_grid_tariffs.substation_prosumers import (
    generate_prosumer_cronian_config,
    create_prosumer_optimization_model,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the entire workflow of the simulation."""
    parser = argparse.ArgumentParser(description="Generate substation prosumer configurations and run optimization.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the main workflow yaml config file.")
    parser.add_argument(
        "--prosumers",
        nargs="*",
        default=None,
        help="List of substation prosumer names to run optimization for. If not provided or set to 'all', "
        "runs for all substations (column names) defined in the substation loads.csv from the config file.",
    )
    parser.add_argument(
        "--dc-power-flow",
        action="store_true",
        help="If set, uses DC power flow instead of AC power flow for the simulation.",
    )
    parser.add_argument(
        "--exclude-hv-power",
        action="store_true",
        help="If set, excludes the power profiles for high voltage substations in the power flow simulation.",
    )
    parser.add_argument(
        "--base-elec-results-dir",
        type=Path,
        default=None,
        help="Directory containing base case power flow results used for making relative comparisons.",
    )
    parser.add_argument(
        "--vol-tou-tariffs", type=Path, default=None, help="Path to the volumetric TOU tariffs [€/MWh] CSV file."
    )
    parser.add_argument("--cap-tariff", type=float, default=None, help="Capacity tariff cost [€/MW-month].")
    parser.add_argument(
        "--cap-tariff-weights",
        type=Path,
        default=None,
        help="Path to the monthly weighting factors CSV file for the cap tariff.",
    )
    args = parser.parse_args()

    # Setup main directories where inputs and results will be saved
    results_dir = Path("results") / args.config.stem / "outputs"
    data_dir = Path("results") / args.config.stem / "inputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load Configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    scenario_params = config["scenario_params"]
    data_sources = config["data_sources"]
    num_snapshots = config["model_params"]["num_snapshots"]

    # Standardize timeseries indices and slice to the correct number of snapshots
    year = scenario_params["year"]
    snapshots = pd.date_range(start=f"{year}-01-01 00:00:00", periods=num_snapshots, freq="h")

    # Process tariff
    # Raise value error if either cap_tariff or cap_tariff_weights_monthly is provided without the other one
    if (args.cap_tariff is None) != (args.cap_tariff_weights is None):
        raise ValueError("Both cap_tariff and cap_tariff_weights must be provided together.")

    vol_tou_tariffs = pd.read_csv(args.vol_tou_tariffs, index_col="snapshots", parse_dates=True) if args.vol_tou_tariffs else None
    cap_tariff = args.cap_tariff if args.cap_tariff else None
    cap_tariff_weights_monthly = pd.read_csv(args.cap_tariff_weights) if args.cap_tariff_weights else None
    
    if args.vol_tou_tariffs is not None:
        plot_vol_tou_tariffs(vol_tou_tariffs, data_dir)
        logger.info("Successfully plotted volumetric TOU tariffs")

    if cap_tariff is not None and cap_tariff_weights_monthly is not None:
        plot_capacity_tariff(cap_tariff, cap_tariff_weights_monthly, year, data_dir)
        logger.info("Successfully plotted capacity network tariff")

    # 1. ============ District heating network inputs preparation, optimization, and results saving/plotting ===========
    # 1.1 DHN inputs preparation
    dhn_inputs_dir = data_dir / "dhn_inputs"
    dhn_inputs_dir.mkdir(exist_ok=True)
    prepare_district_heating_network_inputs(config, snapshots, dhn_inputs_dir)

<<<<<<< HEAD
    # # 1.2 Build and solve the district heating network optimization model
    # solved_lc_network = build_and_solve_least_cost_network(
    #     dhn_inputs_dir, config, vol_tou_tariffs, cap_tariff, cap_tariff_weights_monthly
    # )

    # # 1.3 Save the district heating network optimization model results.
    # dhn_results_dir = results_dir / "dhn_results"
    # dhn_results_dir.mkdir(exist_ok=True)
    # save_network_results(solved_lc_network, dhn_results_dir)

    # # 1.4 Plot the district heating network optimization results.
    # dhn_results_dir_plots = dhn_results_dir / "figures"
    # dhn_results_dir_plots.mkdir(exist_ok=True)
    # plot_dhn_results(solved_lc_network, output=dhn_results_dir_plots)
=======
    # 1.2 Build and solve the district heating network optimization model
    solved_lc_network = build_and_solve_least_cost_network(
        dhn_inputs_dir, config, vol_tou_tariffs, cap_tariff, cap_tariff_weights_monthly
    )

    # 1.3 Save the district heating network optimization model results.
    dhn_results_dir = results_dir / "dhn_results"
    dhn_results_dir.mkdir(exist_ok=True)
    save_network_results(solved_lc_network, dhn_results_dir)

    # 1.4 Plot the district heating network optimization results.
    dhn_results_dir_plots = dhn_results_dir / "figures"
    dhn_results_dir_plots.mkdir(exist_ok=True)
    plot_dhn_results(solved_lc_network, output=dhn_results_dir_plots)
>>>>>>> ba51fe1 ([WIP] implement orchestrator)

    logger.info(" ============ Successfully completed district heating network model runs 🎉🎉🎉 ============ ")

    # 2. ============= Substation prosumers' inputs preparation, optimization, and results saving/plotting =============
    substation_loads_path = fill_path_wildcards(data_sources["electricity_system"]["loads"], scenario_params)
    raw_substation_loads = pd.read_csv(substation_loads_path, index_col=0, parse_dates=True)
    substation_profiles = raw_substation_loads.copy()
    substation_profiles = substation_profiles.iloc[:num_snapshots]
    substation_profiles.index = snapshots
    substation_profiles = split_bidirectional_profiles(substation_profiles)

    if args.prosumers is None or args.prosumers == ["all"]:
        selected_prosumers = list(raw_substation_loads.columns)
    elif len(args.prosumers) == 0:
        selected_prosumers = []
    else:
        selected_prosumers = args.prosumers

    prosumer_configs_dir = data_dir / "prosumer_configs"
    prosumer_configs_dir.mkdir(exist_ok=True)
    solver = pyo.SolverFactory("gurobi")

    prosumers_results_dir = results_dir / "prosumer_results"
    prosumers_results_dir.mkdir(exist_ok=True)

<<<<<<< HEAD
    logger.info(f"Running optimization for prosumers: {selected_prosumers}")
    for prosumer_name in selected_prosumers:
        prosumer = generate_prosumer_cronian_config(
            splitted_substation_profiles=substation_profiles,
            column_name=prosumer_name,
            output=prosumer_configs_dir,
        )["Prosumers"]
=======
    for prosumer_name in selected_prosumers:
        generate_prosumer_cronian_config(
            processed_substation_profiles=substation_profiles,
            column_name=prosumer_name,
            output=prosumer_configs_dir,
        )

        prosumer_config_path = prosumer_configs_dir / f"P0_{prosumer_name}.yaml"
        prosumer = load_prosumer_cronian_config(prosumer_config_path)
>>>>>>> ba51fe1 ([WIP] implement orchestrator)

        model = create_prosumer_optimization_model(
            prosumer=prosumer,
            electricity_price=pd.read_csv(dhn_inputs_dir / "electricity_price.csv", index_col=0, parse_dates=True),
            timeseries_data=substation_profiles,
            vol_tou_tariffs_demand=vol_tou_tariffs,
            cap_tariff=cap_tariff,
            cap_tariff_weights_monthly=cap_tariff_weights_monthly,
        )

        solver.solve(model)

        # Extract and save results for the prosumer
        dispatch = extract_prosumer_dispatch(model, prosumer)
<<<<<<< HEAD

        electric_power_attr_name = f"P0_{prosumer_name}_electric_power"
        electric_power = [-1 * pyo.value(getattr(model, electric_power_attr_name)[t]) for t in model.time]
        dispatch["electric_power"] = electric_power
        dispatch.to_csv(prosumers_results_dir / f"{prosumer_name}_dispatch.csv")

        # if prosumer_name == "Nootdorp2_23":
        #    print(f"Dispatch for {prosumer_name}:")
        #    print(dispatch)
        #    break

        # # TODO: Fix the infeasibility of ``Laagveen_10`` and ``Nootdorp2_23`` prosumers and remove this condition.
        # if prosumer_name not in ["Laagveen_10", "Nootdorp2_23"]:  # These are currently infeasible.
        #     electric_power_attr_name = f"P0_{prosumer_name}_electric_power"
        #     # We reverse the sign such that +ve values are consumption from, and -ve values are injection into the grid.
        #     electric_power = pd.DataFrame({
        #         "snapshots": snapshots,
        #         "electric_power": [-1 * pyo.value(getattr(model, electric_power_attr_name)[t]) for t in model.time]
        #     }).set_index("snapshots")
        #     electric_power.to_csv(prosumers_results_dir / f"{prosumer_name}_electric_power.csv")
    
        logger.info(f" 💯💯 💯 Successfully optimized and saved results for prosumer {prosumer_name}")

    exit()
=======
        dispatch.to_csv(prosumers_results_dir / f"{prosumer_name}_dispatch.csv")

        # TODO: Fix the infeasibility of ``Laagveen_10`` and ``Nootdorp2_23`` prosumers and remove this condition.
        if prosumer_name not in ["Laagveen_10", "Nootdorp2_23"]:  # These are currently infeasible.
            electric_power_attr_name = f"P0_{prosumer_name}_electric_power"
            # We reverse the sign such that +ve values are consumption from, and -ve values are injection into the grid.
            electric_power = pd.DataFrame({
                "snapshots": snapshots,
                "electric_power": [-1 * pyo.value(getattr(model, electric_power_attr_name)[t]) for t in model.time]
            }).set_index("snapshots")
            electric_power.to_csv(prosumers_results_dir / f"{prosumer_name}_electric_power.csv")
    
        logger.info(f"Successfully optimized and saved results for prosumer {prosumer_name}")
>>>>>>> ba51fe1 ([WIP] implement orchestrator)

    # 3. ============ Electric network power flow inputs preparation, simulation, and results saving/plotting ==========
    # Load the base electricity network excel and the links.csv which maps DHN assets to electricity network buses.
    base_electricity_network_path = fill_path_wildcards(data_sources["electricity_system"]["network"], scenario_params)
    all_sheets = pd.read_excel(base_electricity_network_path, sheet_name=None, index_col=0)
    links_df = pd.read_csv(dhn_inputs_dir / "network" / "links.csv")

    # Add DHN assets to the electricity network excel file in their corresponding sheets (load or sgen)
    all_sheets = add_dhn_assets_to_pandapower_excel_sheets(all_sheets, links_df)

    # Save the updated electricity network excel file with DHN assets added to the data_dir.
    elec_inputs_dir = data_dir / "electricity_inputs"
    elec_inputs_dir.mkdir(exist_ok=True)

    electricity_network_with_dhn_assets_path = elec_inputs_dir / "elec-network-with-dhn-assets.xlsx"
    logger.info(f"Saving updated network file to {electricity_network_with_dhn_assets_path}")
    with pd.ExcelWriter(electricity_network_with_dhn_assets_path) as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    logger.info("Successfully saved the updated electricity network file with DHN assets 🎉🎉🎉")

    # Prepare the power profiles for the electricity model, which include the substation loads and the electricity
    # consumption and generation of DHN assets.
    # First, we load the substation load profiles and split them into separate load and sgen profiles.
    logger.info("Preparing substation load profiles...")
    substation_loads_optimized = combine_substation_prosumers_electric_power(selected_prosumers, prosumers_results_dir)

    # Process the substation profiles to split bidirectional profiles into unidirectional load and sgen.
    substation_loads_optimized = split_bidirectional_profiles(substation_loads_optimized)

    # Initialize list of dataframes to process with the substation profiles (profiles of the DHN assets will be added)
    dfs_to_process = [substation_loads_optimized]

    # Next, load electricity consumption and generation of DHN assets from there.
    logger.info("Loading electricity consumption and generation data of the DHN assets...")
    elec_consumption_path = dhn_results_dir / "electricity_consumption.csv"
    elec_generation_path = dhn_results_dir / "electricity_generation.csv"

    electricity_consumption = pd.read_csv(elec_consumption_path, index_col=0)
    electricity_generation = pd.read_csv(elec_generation_path, index_col=0)

    # The column names already match the pandapower element names in the elec-network-with-dhn-assets.xlsx that was done
    # with the add_dhn_assets_to_pandapower_excel_sheets function, so we just append them directly.
    logger.info("Processing and combining dataframes into power_profiles.csv")
    dfs_to_process.extend([electricity_consumption, electricity_generation])

    # Combine all dataframes and standardize (e.g., pandapower only supports integer indices, etc.) for the simulation
    final_profiles_dfs = []
    for df in dfs_to_process:
        df = df.iloc[:num_snapshots, :]
        df.index = range(len(df))
        df.index.name = "snapshots"
        final_profiles_dfs.append(df)

    # Concatenate all dataframes along columns to create the final power profiles and save to CSV.
    power_profiles_dhn_and_substation_prosumers = pd.concat(final_profiles_dfs, axis=1)
    power_profiles_path = elec_inputs_dir / "power_profiles.csv"
    power_profiles_dhn_and_substation_prosumers.to_csv(power_profiles_path)
    logger.info(f"Successfully saved combined power profiles to {power_profiles_path} 🎉🎉🎉")

    # Run the electricity network power flow simulation.
    elec_results_dir = results_dir / "electricity_results"
    elec_results_dir.mkdir(exist_ok=True)

    run_electricity_model(
        config_path=args.config,
        network_path=electricity_network_with_dhn_assets_path,
        profiles_path=power_profiles_path,
        output_dir=elec_results_dir,
        exclude_hv_power=args.exclude_hv_power,
        dc_power_flow=args.dc_power_flow,
    )

    # Plot the electricity network power flow results.
    plot_electricity_results(
        network_path=electricity_network_with_dhn_assets_path,
        elec_results_dir=elec_results_dir,
        experiment_name=args.config.stem,
        base_results_dir=args.base_elec_results_dir,
        output_dir=elec_results_dir,
    )

    # 4. ============ Finally, make and save a copy of the config file in the data_dir for reproducibility =============
    try:
        shutil.copy(args.config, data_dir)
        logger.info(f" Copied workflow config {args.config.name} to {data_dir} for reproducibility.")
    except Exception as e:
        logger.error(f"Could not copy workflow config file {args.config}. Error: {e}")

    logger.info(" 🎉🎉🎉 Done! ... Successfully completed the entire workflow 🎉🎉🎉 ")


def prepare_district_heating_network_inputs(config: dict, snapshots: pd.DatetimeIndex, output_dir: Path) -> None:
    """Prepares the input data for the district heating network model based on the provided configuration."""
    # Prepare csv data files
    processed_dfs = prepare_and_save_heat_model_csv_data(
        data_sources=config["data_sources"],
        scenario_params=config["scenario_params"],
        adjustments=config.get("scenario_adjustments", {}),
        snapshots=snapshots,
        output_dir=output_dir,
    )

    # Prepare network component CSV files
    prepare_network_component_files(
        data_sources=config["data_sources"],
        scenario_params=config["scenario_params"],
        adjustments=config.get("scenario_adjustments", {}),
        output_dir=output_dir,
    )

    # Generate heat input data plots for quick diagnostics
    figure_folder = output_dir / "figures"
    figure_folder.mkdir(exist_ok=True)
    plot_heat_input_data(processed_dfs, figure_folder)

    logger.info(" =========== Successfully prepared and plotted district heating network inputs 🎉🎉🎉 =========== ")


def load_prosumer_cronian_config(path) -> dict:
    """Loads the cronian configuration for a substation prosumer from a yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)["Prosumers"]


def combine_substation_prosumers_electric_power(prosumer_names: list, prosumer_results_dir: Path) -> pd.DataFrame:
    """Loads the electric power profile dateframe for each substation prosumer and combines them.
    
    Args:
    -----
        prosumer_names: List of prosumer (substation) names whose electric power
            profiles we want to combine.
        prosumer_results_dir: Directory containing the electric power profile CSV files for each prosumer.
        
    Returns:
    --------
        A single dataframe combining the electric power profiles of the specified
            prosumers (substation), with columns named as {prosumer_name}.
    """
    combined_profiles = []
    for prosumer_name in prosumer_names:
<<<<<<< HEAD
        profile_path = prosumer_results_dir / f"{prosumer_name}_dispatch.csv"
        profile_df = pd.read_csv(profile_path, index_col=0, parse_dates=True)
        electric_power_df = profile_df[["electric_power"]].copy()
        electric_power_df.rename(columns={"electric_power": prosumer_name}, inplace=True)
        combined_profiles.append(electric_power_df)
=======
        profile_path = prosumer_results_dir / f"{prosumer_name}_electric_power.csv"
        profile_df = pd.read_csv(profile_path, index_col=0, parse_dates=True)
        profile_df.rename(columns={"electric_power": prosumer_name}, inplace=True)
        combined_profiles.append(profile_df)
>>>>>>> ba51fe1 ([WIP] implement orchestrator)

    combined_profiles_df = pd.concat(combined_profiles, axis=1)

    return combined_profiles_df


if __name__ == "__main__":
    main()
