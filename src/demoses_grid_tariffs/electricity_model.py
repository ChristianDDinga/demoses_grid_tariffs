import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandapower import from_excel as create_network_from_excel
from pandapower import rundcpp, runpp
from pandapower.auxiliary import LoadflowNotConverged, pandapowerNet
from pandapower.diagnostic import diagnostic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_power_flow(
    network_path: Path, power_profiles: pd.DataFrame, snapshots: pd.DatetimeIndex, dc_power_flow: bool, output: Path
) -> None:
    """Run a timeseries power flow calculation.

    Args:
    -----
        network_path: Path to the Excel file containing the network data.
        power_profiles: DataFrame containing the timeseries power profiles for sgen and load elements.
        snapshots: DatetimeIndex which will be used to re-index the results dataframes.
        dc_power_flow: If True, runs DC power flow using rundcpp(), else runs full AC power flow using runpp().
        output: Folder to save the output files.
    """
    if len(power_profiles) != len(snapshots):
        raise ValueError(
            f"Length of power_profiles ({len(power_profiles)}) does not match length of snapshots ({len(snapshots)})."
        )

    # Perform consistency checks between the network and power profiles
    check_electricity_network_consistency(network_path, power_profiles)

    # Create the pandapower network from the excel file
    net = create_network_from_excel(network_path)

    logger.info("🔎🔎🔎========================= Running Initial Diagnostics =========================")
    diagnostic(net, report_style="detailed", warnings_only=False)
    logger.info("🔎🔎🔎======================== Completed Initial Diagnostics ========================")

    # Initialize dictionaries to store results and diagnostics
    results = {
        "bus_vm_pu": [],
        "bus_va_degree": [],
        "line_loading_percent": [],
        "trafo_loading_percent": [],
        "ext_grid_q_mvar": [],
        "ext_grid_p_mw": [],
    }
    diagnostic_info = {}

    # Pre-create zero-filled placeholders to avoid recreating them in the loop
    zero_placeholders = {
        "bus": pd.Series(0.0, index=net.bus.index),
        "line": pd.Series(0.0, index=net.line.index),
        "trafo": pd.Series(0.0, index=net.trafo.index),
        "ext_grid": pd.Series(0.0, index=net.ext_grid.index),
    }

    # Set a seed for reproducibility of random voltage magnitudes for non-converged cases
    np.random.seed(0)

    # Choose the power flow function based on dc_power_flow flag
    power_flow_function = rundcpp if dc_power_flow else runpp

    # Get the list of sgen and load elements from the power profiles columns
    sgen_names = [col for col in power_profiles.columns if col in net.sgen.name.values]
    load_names = [col for col in power_profiles.columns if col in net.load.name.values]
    sgen_index_name_mapping = pd.Series(net.sgen.index, index=net.sgen.name)
    load_index_name_mapping = pd.Series(net.load.index, index=net.load.name)

    # Run the power flow for each time step
    time_steps = range(len(power_profiles))

    for time_step in time_steps:
        current_profiles = power_profiles.iloc[time_step]
        net.sgen.loc[sgen_index_name_mapping[sgen_names], 'p_mw'] = current_profiles[sgen_names].values
        net.load.loc[load_index_name_mapping[load_names], 'p_mw'] = current_profiles[load_names].values

        try:
            power_flow_function(net)
            collect_results(net, results)
        except LoadflowNotConverged:
            # diag_results = run_and_store_diagnostics(net, time_step, diagnostic_info)

            if not dc_power_flow:  # AC failed, run DC as fallback
                logger.warning(f"AC power flow failed at timestep {time_step}. Running DC power flow as fallback.")
                try:
                    rundcpp(net)
                    collect_results(net, results)
                    logger.info(f"✅ DC power flow succeeded at timestep {time_step} as fallback ✅")
                except LoadflowNotConverged:
                    logger.error(f"Both AC and DC power flow failed at timestep {time_step}. Filling with zeros.")
                    append_zero_results(results, zero_placeholders)

                # # Overwrite bus voltages based on diagnostic results
                # overload_info = diag_results.get("overload", {})
                # if overload_info.get("load", False):
                #     logger.info(f"Non-convergence at {time_step} due to load overload. Setting random under-voltage.")
                #     voltages = pd.Series(np.random.uniform(MIN_VOLTAGE, 0.9, size=len(net.bus)), index=net.bus.index)
                #     results["bus_vm_pu"][-1] = voltages  # Replace the last entry
                # elif overload_info.get("generation", False):
                #     logger.info(f"Non-convergence at {time_step} due to gen overload. Setting random over-voltage.")
                #     voltages = pd.Series(np.random.uniform(1.1, MAX_VOLTAGE, size=len(net.bus)), index=net.bus.index)
                #     results["bus_vm_pu"][-1] = voltages  # Replace the last entry

            else:  # DC-only mode failed
                logger.error(f"DC power flow failed at timestep {time_step}. Filling with zeros.")
                append_zero_results(results, zero_placeholders)

    # Writeout diagnostics results
    if diagnostic_info:
        diag_file = output / "non_converged_diagnostics.json"
        with open(diag_file, "w") as f:
            json.dump(diagnostic_info, f, indent=4, default=str)
        logger.info(f"Saved diagnostics for non-converged timesteps to {diag_file}")

    # Save the results to CSV files
    save_results_to_csv(results, output)

    # Create copies of results with named columns and datetime index
    create_a_copy_of_results_with_named_columns_and_indices(net, snapshots, output)


def save_results_to_csv(results: dict, output: Path) -> None:
    """Combines lists of pandapower result series into DataFrames and saves them to CSV files.

    Args:
    -----
        results: A dict with keys as pandapower result variable names (e.g., 'bus_vm_pu') and
            values are lists of pandas Series, each representing the value at a timestep.
        output: Path where the CSV files will be saved.
    """
    for key, data_list in results.items():
        if not data_list:
            logger.warning(f"No results to save for {key}, as there is no data. Skipping.")
            continue

        # Combine the list of Series into a single DataFrame. Each series becomes a row.
        result_df = pd.DataFrame(data_list)

        # Drop the index to which is the result variable name (e.g., 'vm_pu') to be replaced by default integer index
        result_df = result_df.reset_index(drop=True)

        # Create the correct output directory (e.g., 'res_bus/vm_pu.csv') and save the DataFrame as a CSV file
        try:
            component, variable_ext = key.split('_', 1)
            filename = f"{variable_ext}.csv"
            component_dir = output / f"res_{component}"
            component_dir.mkdir(parents=True, exist_ok=True)
            full_path = component_dir / filename
            result_df.to_csv(full_path)
        except ValueError:
            logger.error(f"Could not parse key '{key}' into component and variable. Skipping.")


def create_a_copy_of_results_with_named_columns_and_indices(
    net: pandapowerNet, snapshots: pd.DatetimeIndex, output: Path
) -> None:
    """Create a copy of the results files with named columns and indices.

    Pandapower only allows dataframes indices to be integers, while it uses component
    indices (instead of names) as columns when writing out results. This function
    creates copies of the results CSV files with named columns and a datetime index.

    Args:
    -----
        net: pandapower network.
        snapshots: DatetimeIndex which used to re-index the results dataframes.
        output: Folder where the output files are saved.
    """
    # Create a mapping from indices to names for each component type
    index_to_name_mapping_of_component_types = {
        "bus": dict(zip([str(i) for i in net.bus.index], net.bus["name"])),
        "line": dict(zip([str(i) for i in net.line.index], net.line["name"])),
        "trafo": dict(zip([str(i) for i in net.trafo.index], net.trafo["name"])),
    }

    # Define which result files to process for each component type
    component_res_files_mapping = {
        "bus": ["vm_pu.csv", "va_degree.csv"],
        "line": ["loading_percent.csv"],
        "trafo": ["loading_percent.csv"],
    }

    for component, index_to_name_mapping in index_to_name_mapping_of_component_types.items():
        res_files = component_res_files_mapping[component]

        for filename in res_files:
            result_file = output / f"res_{component}" / filename

            if result_file.exists():
                results_df = pd.read_csv(result_file, index_col=0)
                results_df.rename(columns=index_to_name_mapping, inplace=True)  # Rename columns using the mapping

                if len(results_df) != len(snapshots):
                    raise ValueError(
                        f"Length of results ({len(results_df)}) does not match length of snapshots ({len(snapshots)})."
                    )

                # Set the index to the provided snapshots
                results_df.index = snapshots
                results_df.index.name = "snapshot"

                # Save the modified dataframe to a new CSV file
                new_filename = filename.replace(".csv", "_with_names.csv")
                results_df_with_names_file = output / f"res_{component}" / new_filename
                results_df.to_csv(results_df_with_names_file)

            else:
                logger.warning(f"Result file {result_file} does not exist. Skipping.")

    logger.info(" Successfully created copies of results files with named columns and datetime index.")


def check_electricity_network_consistency(network_path: Path, power_profiles: pd.DataFrame) -> None:
    """Perform consistency checks between the pandapower network and the power profiles."""
    # 1. Check that all columns in the power profiles are listed in the load or sgen sheets of the pandapower network.
    sgen_df = pd.read_excel(network_path, sheet_name="sgen", index_col=0)
    load_df = pd.read_excel(network_path, sheet_name="load", index_col=0)

    sgen_names = set(sgen_df["name"].tolist())
    load_names = set(load_df["name"].tolist())

    # Check that all columns in power_profiles are in either sgen_names or load_names
    for column in power_profiles.columns:
        if column not in sgen_names and column not in load_names:
            raise ValueError(
                f"Column '{column}' in power profiles is not in sgen or load sheets of the pandapower network."
            )

    # 2. Also check that there are no NaN values in the power profiles
    if power_profiles.isnull().values.any():
        raise ValueError("Power profiles contain NaN values. Please ensure all profiles are complete.")

    logger.info("✅✅✅ Electricity network consistency check passed: all power profile columns are valid.")


def run_and_store_diagnostics(net: pandapowerNet, time_step: int, diag_info_dict: dict) -> dict:
    """Runs pandapower diagnostics and stores the results in a dictionary."""
    logger.info(f"🔎🔎🔎 Running diagnostics for timestep {time_step}...")
    diag_results = diagnostic(net, report_style="detailed", warnings_only=False, return_result_dict=True)
    diag_info_dict[f"timestep_{time_step}"] = diag_results
    logger.info(f"🔎🔎🔎 Diagnostics for timestep {time_step} complete.")
    return diag_results


def collect_results(net: pandapowerNet, results_dict: dict) -> None:
    """Appends the current simulation results from the net object to the results dictionary."""
    results_dict["bus_vm_pu"].append(net.res_bus.vm_pu.copy())
    results_dict["bus_va_degree"].append(net.res_bus.va_degree.copy())
    results_dict["line_loading_percent"].append(net.res_line.loading_percent.copy())
    results_dict["trafo_loading_percent"].append(net.res_trafo.loading_percent.copy())
    results_dict["ext_grid_q_mvar"].append(net.res_ext_grid.q_mvar.copy())
    results_dict["ext_grid_p_mw"].append(net.res_ext_grid.p_mw.copy())


def append_zero_results(results_dict: dict, zero_placeholders: dict) -> None:
    """Appends pre-defined zero-filled Series to the results dictionary."""
    results_dict["bus_vm_pu"].append(zero_placeholders["bus"])
    results_dict["bus_va_degree"].append(zero_placeholders["bus"])
    results_dict["line_loading_percent"].append(zero_placeholders["line"])
    results_dict["trafo_loading_percent"].append(zero_placeholders["trafo"])
    results_dict["ext_grid_q_mvar"].append(zero_placeholders["ext_grid"])
    results_dict["ext_grid_p_mw"].append(zero_placeholders["ext_grid"])
