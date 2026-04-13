import calendar
import logging
import shutil
from pathlib import Path

import pandas as pd

from demoses_grid_tariffs.helper_functions import fill_path_wildcards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYPSA_COMPONENTS_TO_PROCESS = ["Link", "Generator", "StorageUnit", "Store"]

# Sheets in the excel file to be converted to CSVs describing the pypsa heat network.
EXCEL_SHEETS = ["buses", "carriers", "generators", "links", "loads", "storage_units", "stores"]

NUM_DECIMALS = 5  # Number of decimal places to round numeric values to avoid numerical issues with solvers.


def prepare_and_save_heat_model_csv_data(
    data_sources: dict, scenario_params: dict, adjustments: dict, snapshots: pd.DatetimeIndex, output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Orchestrates the loading, processing, and saving of all heat system dataframes.

    Args:
    -----
        data_sources: The 'data_sources' section of the workflow yaml config.
        scenario_params: The 'scenario_params' for filling path wildcards.
        adjustments: The 'scenario_adjustments' section of the workflow yaml config.
        snapshots: The set of timestamps to consider in the optimization.
        output_dir: The base output directory for the prepared inputs (e.g., .../01_heat_inputs).

    Returns:
    --------
        A dictionary of processed, ready-to-save dataframes.
    """
    # Step 1: Load all raw data from disk into memory
    raw_dataframes = load_raw_heat_data(data_sources, scenario_params)

    # Step 2: Apply all modifications (scaling, indexing, slicing)
    processed_dfs = process_heat_model_data(raw_dataframes, adjustments, snapshots)

    # Step 3: Save the processed dataframes to the output directory
    for filename, file_df in processed_dfs.items():
        file_path = output_dir / f"{filename}.csv"
        file_df.to_csv(file_path, index=True)

    logger.info(f"Successfully saved all prepared dataframes to {output_dir}")

    return processed_dfs


def load_raw_heat_data(data_sources: dict, scenario_params: dict) -> dict[str, pd.DataFrame]:
    """Loads all raw heat system dataframes from the paths specified in the config.

    Args:
    -----
        data_sources: The 'data_sources' section of the config dictionary.
        scenario_params: The 'scenario_params' for filling path wildcards.

    Returns:
    --------
        A dictionary mapping data names (e.g., "heat_demand") to their loaded DataFrames.
    """
    logger.info("Loading raw heat data...")
    heat_data_paths = data_sources["heat_system"]

    try:
        raw_data = {
            "temperature": pd.read_csv(
                fill_path_wildcards(heat_data_paths["temperature"], scenario_params),
                index_col=0,
                parse_dates=True,
            ),
            "demand": pd.read_csv(
                fill_path_wildcards(heat_data_paths["demand"], scenario_params), index_col=0, parse_dates=True,
            ),
            "etm_heat_network_profiles": pd.read_csv(
                fill_path_wildcards(heat_data_paths["etm_heat_network_profiles"], scenario_params),
                index_col=0,
                parse_dates=True,
            ),
            "hydrogen_price": pd.read_csv(
                fill_path_wildcards(heat_data_paths["hydrogen_price"], scenario_params),
                index_col=0,
                parse_dates=True,
            ),
            "electricity_price": pd.read_csv(
                fill_path_wildcards(heat_data_paths["electricity_price"], scenario_params),
                index_col=0,
                parse_dates=True,
            ),
            "solar_availability": pd.read_csv(
                fill_path_wildcards(heat_data_paths["solar_availability"], scenario_params),
                index_col=0,
                parse_dates=True,
            ),
            "static_prices": pd.read_csv(
                fill_path_wildcards(heat_data_paths["static_prices"], scenario_params), index_col=0,
            ),
        }

        logger.info("Successfully loaded all raw data files.")

    except FileNotFoundError as e:
        logger.error(f"Error loading heat system data file: {e}")
        raise

    # For electricity, and hydrogen dfs, drop every other column  except first two
    for key in ["electricity_price", "hydrogen_price", "solar_availability"]:
        df = raw_data[key]
        if df.shape[1] > 2:
            df = df.iloc[:, :1]
            logger.warning(
                f"Dataframe '{key}' has more than 2 columns. Only the first two columns will be kept.",
            )
        raw_data[key] = df

    return raw_data


def process_heat_model_data(
    raw_data: dict[str, pd.DataFrame], adjustments: dict, snapshots: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Applies in-memory modifications to the raw dataframes of the heat system model.

    Args:
    -----
        raw_data: Dictionary of raw dataframes from the `load_raw_heat_data` function.
        adjustments: Contains optional modification factors.
        snapshots: The set of timestamps to consider in the optimization.

    Returns:
    --------
        A dictionary of processed, ready-to-save dataframes.
    """
    logger.info("Applying scenario adjustments and processing data...")

    processed_data = raw_data.copy()

    # Apply heat demand settings if specified
    if "heat_demand_settings" in adjustments:
        factors = adjustments["heat_demand_settings"]
        for region, factor in factors.items():
            processed_data["demand"].loc[:, region] *= factor
            logger.info(f"Scaled heat demand in {region} by a factor of {factor}.")

    # Apply normalization using ETM profiles if specified
    if adjustments.get("apply_normalization_using_etm_profiles", False):
        etm_profiles_df = processed_data.get("etm_heat_network_profiles")
        if etm_profiles_df is None:
            raise ValueError("ETM heat network profiles data is missing, cannot apply normalization.")

        # Read the common normalization profile from the `heat demand built environment normalized` column
        normalization_profile = etm_profiles_df["heat demand built environment normalized"].values

        # Ensure min and max of normalization profile are 0 and 1
        norm_min = normalization_profile.min()
        norm_max = normalization_profile.max()
        if not (0 <= norm_min <= 1) or not (0 <= norm_max <= 1):
            raise ValueError(
                f"ETM normalization profile values must be between 0 and 1. Found min: {norm_min}, max: {norm_max}.",
            )

        demand_df = processed_data["demand"].copy()

        for region in demand_df.columns:
            region_demand = demand_df[region].values

            if len(region_demand) != len(normalization_profile):
                raise ValueError(
                    f"Length mismatch between demand data and ETM normalization profile for region '{region}'.",
                )

            region_max = region_demand.max()
            if region_max == 0:
                logger.warning(f"Maximum demand for region '{region}' is zero, skipping normalization.")
                continue

            # Normalize the region's demand using the ETM profile scaled to the region's max demand
            demand_df[region] = normalization_profile * region_max
            logger.info(f"Normalized heat demand for region '{region}' using ETM profiles.")

        processed_data["demand"] = demand_df

    # First build the static prices df only for the target year without any other modifications
    static_df = raw_data["static_prices"].copy()
    target_year = snapshots[0].year
    static_df_to_modify = pd.DataFrame(static_df.loc[[target_year]])
    processed_data["static_prices"] = static_df_to_modify  # Update the processed data immediately

    # Apply static prices settings if specified
    if "static_prices_settings" in adjustments:
        carrier_price_dict = adjustments["static_prices_settings"]
        static_df_modified = processed_data["static_prices"].copy()
        for carrier, new_price in carrier_price_dict.items():
            if carrier not in static_df_modified.columns:
                raise ValueError(f"Carrier '{carrier}' from config not found in static_prices data.")

            static_df_modified.loc[target_year, carrier] = float(new_price)
            logger.info(f"Set static price of '{carrier}' to {float(new_price)}.")

        processed_data["static_prices"] = static_df_modified

    # Process timeseries data
    for key in processed_data.keys():
        if key == "static_prices":
            continue  # Static prices already handled above

        # Slice to the correct number of snapshots and set index
        df = processed_data[key].copy()
        df = df.iloc[: len(snapshots)]
        df.index = snapshots
        df.index.name = "snapshots"

        # Scale electricity prices if specified
        if key == "electricity_price" and "electricity_price_settings" in adjustments:
            factor = adjustments["electricity_price_settings"]["scaling_factor"]
            max_price = adjustments["electricity_price_settings"]["max_price"]
            df *= factor
            df = df.clip(upper=max_price)
            logger.info(f"Scaled electricity prices by a factor of {factor} with {max_price=}.")

        # Scale hydrogen prices if specified
        if key == "hydrogen_price" and "hydrogen_price_settings" in adjustments:
            factor = adjustments["hydrogen_price_settings"]["scaling_factor"]
            max_price = adjustments["hydrogen_price_settings"]["max_price"]
            df *= factor
            df = df.clip(upper=max_price)
            logger.info(f"Scaled hydrogen prices by a factor of {factor} with {max_price=}.")

        # Scale solar availability if specified
        if key == "solar_availability" and "solar_thermal_settings" in adjustments:
            factor = adjustments["solar_thermal_settings"]["scaling_factor"]
            max_capacity = adjustments["solar_thermal_settings"]["max_capacity_factor"]
            df *= factor
            df = df.clip(upper=max_capacity)
            logger.info(f"Scaled solar thermal capacity factor by a factor of {factor} with {max_capacity=}.")

        # Increase/decrease temperature if specified
        if key == "temperature" and "temperature_settings" in adjustments:
            increase_degree = adjustments["temperature_settings"]["increase_degree"]
            df += increase_degree
            min_temp = adjustments["temperature_settings"]["min_temp"]
            max_temp = adjustments["temperature_settings"]["max_temp"]
            df = df.clip(upper=max_temp, lower=min_temp)
            logger.info(f"Increased temperature by {increase_degree} degrees Celsius with {min_temp=}, {max_temp=}.")

        processed_data[key] = df

    logger.info("Successfully completed heat system data processing.")

    # Correct summer heat demand for specific regions if specified
    if "summer_period_demand_correction" in adjustments:
        summer_settings = adjustments["summer_period_demand_correction"]
        processed_data = correct_summer_heat_demand(
            processed_data,
            summer_settings["months"],
            summer_settings["decrease_factor"],
        )

    # Round all numeric values
    for key, df in processed_data.items():
        processed_data[key] = df.round(NUM_DECIMALS)

    return processed_data


def correct_summer_heat_demand(
    processed_data: dict[str, pd.DataFrame],
    summer_months_to_correct: list[str],
    demand_decrease_factors: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """Corrects unrealistic heat demand during specified months for a list of regions.

    Args:
    -----
        processed_data: Dictionary of dataframes, must include 'demand'.
        summer_months_to_correct: A list of month names (e.g., ["July", "August"]).
        demand_decrease_factors: A dictionary mapping region names to their demand decrease factors.

    Returns:
    --------
        The dictionary with the modified demand dataframe.
    """
    if "demand" not in processed_data:
        raise ValueError("The 'demand' dataframe is missing from processed_data.")

    demand_df = processed_data["demand"].copy()

    if not isinstance(demand_df.index, pd.DatetimeIndex):
        raise TypeError("The index of the 'demand' dataframe must be a DatetimeIndex.")


    # calendar.month_name is a list where index 1 is 'January', 2 is 'February', etc.
    month_map = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}

    try:
        # Convert list of month names (from config) to list of month numbers
        month_numbers = [month_map[month.lower()] for month in summer_months_to_correct]
    except KeyError as e:
        raise ValueError(f"Invalid month name in summer_months_to_correct: {e}")

    summer_mask = demand_df.index.month.isin(month_numbers)

    # If no data for the specified months, skip correction
    if not summer_mask.any():
        corrected_months_str = ", ".join(summer_months_to_correct)
        logger.info(f"No data for the specified months ({corrected_months_str}) found. Skipping correction.")
        return processed_data

    for region, decrease_factor in demand_decrease_factors.items():
        if region not in demand_df.columns:
            raise ValueError(f"Region '{region}' not found in demand dataframe columns.")

        # Apply the demand decrease factor to the specified summer months
        logger.info(
            f"Correcting heat demand for '{region}' during {', '.join(summer_months_to_correct)} "
            f"with decrease factor: {decrease_factor:.2f}.",
        )
        demand_df.loc[summer_mask, region] *= decrease_factor

    processed_data["demand"] = demand_df

    return processed_data


def prepare_network_component_files(
    data_sources: dict, scenario_params: dict, adjustments: dict, output_dir: Path,
) -> None:
    """Orchestrates the copying and modification of all pypsa network component CSV files."""
    # Step 1: Copy the entire heat network directory to the output directory
    copied_network_path = copy_files_describing_pypsa_heat_network(data_sources, scenario_params, output_dir)

    # Step 2: Loop through specified components and modify their CSVs based on the adjustments provided
    for component in PYPSA_COMPONENTS_TO_PROCESS:
        modify_pypsa_component_csv(component, copied_network_path, adjustments)

    logger.info(f"Successfully prepared all network component files in {output_dir}")


def copy_files_describing_pypsa_heat_network(data_sources: dict, scenario_params: dict, output_dir: Path) -> Path:
    """Copies files describing the PyPSA heat network to the scenario's input folder.

    Args:
    -----
        data_sources: The 'data_sources' section of the global (workflow) config.
        scenario_params: The 'scenario_params' for filling path wildcards.
        output_dir: The base output directory for the prepared inputs (e.g., .../01_heat_inputs).

    Returns:
    --------
        The path to the newly created network directory in the destination.
    """
    heat_network_path = fill_path_wildcards(data_sources["heat_system"]["network"], scenario_params)
    if not heat_network_path.exists():
        raise FileNotFoundError(f"Heat network file or directory not found at: {heat_network_path}")

    logger.info("Copying pypsa (heat) network file or directory ...")

    network_destination_dir = output_dir / "network"

    # If heat_network_path Excel file convert CSVs first, else copy directly
    if heat_network_path.suffix in [".xls", ".xlsx", ".xlsm"]:
        network_destination_dir.mkdir(parents=True, exist_ok=False)
        generate_pypsa_network_csvs_from_excel(heat_network_path, EXCEL_SHEETS, network_destination_dir)
    else:
        shutil.copytree(heat_network_path, network_destination_dir)
        logger.info(f"Copied from '{heat_network_path}' to '{network_destination_dir}'.")

    return network_destination_dir


def generate_pypsa_network_csvs_from_excel(excel_file_path: Path, sheet_names: list[str], output_dir: Path) -> None:
    """Generates PyPSA network CSV files from an Excel file.

    Args:
    -----
        excel_file_path: The path to the Excel file containing network data.
        sheet_names: The names of the sheets to be converted to CSV files.
        output_dir: The directory where the output CSV files will be saved.
    """
    if not excel_file_path.exists():
        raise FileNotFoundError(f"Excel file not found at: {excel_file_path}")

    for sheet in [s.lower() for s in sheet_names]:
        try:
            df = pd.read_excel(excel_file_path, sheet_name=sheet)

            # Drop unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            # Remove everything after an entire empty row (inclusive)
            if df.isnull().all(axis=1).any():
                first_empty_row_idx = df.index[df.isnull().all(axis=1)][0]
                df = df.loc[: first_empty_row_idx - 1, :]

            # Buses: drop the 'actual_node_in_the_heat_grid' column if it exists
            if sheet == "buses" and "actual_node_in_the_heat_grid" in df.columns:
                df = df.drop(columns=["actual_node_in_the_heat_grid"])

            # Round all numeric values to NUM_DECIMALS to avoid numerical issues with solvers.
            df = df.round(NUM_DECIMALS)

            df.to_csv(output_dir / f"{sheet}.csv", index=False)

        except Exception as e:
            logger.error(f"Error processing sheet '{sheet}': {e}")
            raise


def modify_pypsa_component_csv(component: str, network_csv_path: Path, adjustments: dict) -> None:
    """Loads the copied component.csv file from the network_csv_path and overwrites their p_nom_max.

    Args:
    -----
        component: The type of the PyPSA component (e.g., "Link", "Generator").
        network_csv_path: The path to the destination network folder (returned by copy_network_directory).
        adjustments: The 'scenario_adjustments' section of the global (workflow) config.
    """
    if component not in PYPSA_COMPONENTS_TO_PROCESS:
        raise ValueError(f"'{component=}' is not supported for modification. Supported: {PYPSA_COMPONENTS_TO_PROCESS}")

    # Determine the correct CSV file name based on the component type
    if component == "StorageUnit":
        component_csv_name = "storage_units.csv"
    else:
        component_csv_name = component.lower() + "s.csv"  # For example: Link -> links.csv

    component_path = network_csv_path / component_csv_name

    if not component_path.exists():
        logger.warning(
            f"Could not find '{component_csv_name}' at {component_path} to overwrite p_nom_max, skipping ...",
        )
        return

    all_value_settings = adjustments.get("assets_value_settings", {})
    component_value_setting = all_value_settings.get(component, {})

    # Only proceed if there is actually something to do.
    if not component_value_setting:
        return

    # Load, modify, and save the component csv file
    logger.info(f"Applying modifications to the copied {component_csv_name} ...")
    component_df = pd.read_csv(component_path)
    modified_component_df = apply_attribute_adjustments(component, component_df, component_value_setting)
    modified_component_df.to_csv(component_path, index=False)
    logger.info(f"Successfully modified and saved '{component_csv_name}'")


def apply_attribute_adjustments(component: str, component_df: pd.DataFrame, value_settings: dict) -> pd.DataFrame:
    """Applies changes to attributes of assets in the component_df based on the provided value_settings.

    Args:
    -----
        component: The type of the PyPSA component (e.g., "Link", "Generator").
        component_df: The DataFrame containing component data.
        value_settings: Nested dict mapping attributes to asset names to absolute values.
            For example: {'p_nom_max': {'pipe_1': 500}, 'efficiency': {'denhaag_air_heatpump': 3.0}, ...}

    Returns:
    --------
        The modified DataFrame with updated attributes for the specified assets.
    """
    modified_logs = []
    component_df = component_df.set_index(component)

    for  attribute, asset_info in value_settings.items():
        if attribute not in component_df.columns:
            raise ValueError(f"Attribute '{attribute}' from config not found in {component}.csv.")

        for asset, new_val in asset_info.items():
            if asset not in component_df.index:
                raise ValueError(f"Asset '{asset}' from workflow config not found in {component}.csv.")

            component_df.loc[asset, attribute] = float(new_val)
            modified_logs.append(f"{asset=} {attribute=} set to {float(new_val)}")

    if modified_logs:
        logger.info(f"Applied adjustments. Changes: {', '.join(modified_logs)}")
    else:
        logger.info("No new assets were modified based on the provided configuration.")

    return component_df.reset_index()
