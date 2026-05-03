import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from demoses_grid_tariffs.helper_functions import (
    CARRIERS_ELEC_CONS_LINKS,
    CARRIERS_ELEC_PROD_LINKS,
    fill_path_wildcards,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Prepares inputs required to run the powerflow simulation."""
    parser = argparse.ArgumentParser(description="Prepare inputs for the electricity power flow model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the main workflow_config.yaml file.")
    parser.add_argument(
        "--heat-results-dir",
        type=Path,
        help="Path to a heat results directory. Required if using --generate-power-profiles-only.",
    )
    parser.add_argument(
        "--heat-links-csv-path",
        type=Path,
        help="Path to the PyPSA links.csv file for adding DHN assets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path where the prepared electricity model inputs will be saved.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preparing electricity inputs in: {args.output_dir}")

    # Load workflow configuration
    with open(args.config, "r") as f:
        config_file = yaml.safe_load(f)
    scenario_params = config_file["scenario_params"]
    data_sources = config_file["data_sources"]
    num_snapshots = config_file["model_params"]["num_snapshots"]

    # Load the base electricity network excel and the links.csv which maps DHN assets to electricity network buses.
    base_electricity_network_path = fill_path_wildcards(data_sources["electricity_system"]["network"], scenario_params)
    all_sheets = pd.read_excel(base_electricity_network_path, sheet_name=None, index_col=0)
    links_df = pd.read_csv(args.heat_links_csv_path)

    # Add DHN assets to the electricity network excel file in their corresponding sheets (load or sgen)
    all_sheets = add_dhn_assets_to_pandapower_excel_sheets(all_sheets, links_df)

    # Save the modified base electricity network excel file network with DHN assets to a new excel file.
    destination_network_path = args.output_dir / "elec-network-with-dhn-assets.xlsx"
    logger.info(f"Saving updated network file to {destination_network_path}")
    with pd.ExcelWriter(destination_network_path) as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    logger.info(f"Successfully saved updated network file with DHN assets to {destination_network_path} 🎉🎉🎉")

    # Prepare the power profiles for the electricity model, which include the substation loads and the electricity
    # consumption and generation of DHN assets.
    # First, we load the substation load profiles and split them into separate load and sgen profiles.
    logger.info("Preparing substation load profiles...")
    substation_loads_path = fill_path_wildcards(data_sources["electricity_system"]["loads"], scenario_params)
    raw_substation_loads = pd.read_csv(substation_loads_path, index_col=0)
    logger.info(f"Loaded {substation_loads_path}")
    
    # Validate substation profiles length
    if len(raw_substation_loads) < num_snapshots:
        raise ValueError(
            f"Substation_loads must have at least {num_snapshots} snapshots as specified in the config file, "
            f"but it has only {len(raw_substation_loads)} snapshots."
        )

    # Process the substation profiles to split bidirectional profiles into unidirectional load and sgen.
    processed_substation_profiles = split_bidirectional_profiles(raw_substation_loads)
    power_profiles = processed_substation_profiles

    # Initialize list of dataframes to process with the substation profiles (profiles of the DHN assets will be added)
    dfs_to_process = [power_profiles]

    # Next, load electricity consumption and generation of DHN assets from there.
    logger.info("Loading electricity consumption and generation data from heat model...")
    elec_consumption_path = args.heat_results_dir / "electricity_consumption.csv"
    elec_generation_path = args.heat_results_dir / "electricity_generation.csv"

    electricity_consumption = pd.read_csv(elec_consumption_path, index_col=0)
    electricity_generation = pd.read_csv(elec_generation_path, index_col=0)

    logger.info(f"Loaded {elec_consumption_path} and {elec_generation_path}")

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
    power_profiles = pd.concat(final_profiles_dfs, axis=1)
    output_power_profiles_path = args.output_dir / "power_profiles.csv"
    power_profiles.to_csv(output_power_profiles_path)
    logger.info(f"Successfully saved combined power profiles to {output_power_profiles_path} 🎉🎉🎉")


def split_bidirectional_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Splits each bidirectional col (with +ve and -ve values) into separate unidirectional profiles.

    This is done because pandapower requires that all power values be +ve. So, for each column 'colX',
    this function creates two new columns (while dropping the original column):
    - 'colX_load': Contains only the positive values from the original column.
    - 'colX_sgen': Contains the absolute values of the negative values from the original column.

    Args:
    -----
        df: The DataFrame containing profiles of substations with potential bidirectional values.

    Returns:
    --------
        A new DataFrame with separate, unidirectional '_load' and '_sgen' columns per original column.
    """
    logger.info("Splitting bidirectional profiles into unidirectional load and sgen columns...")
    # Keep only positive values for loads, set negatives to 0
    loads = df.clip(lower=0).add_suffix('_load').round(5)
    # Keep only negative values for sgens, and make them positive by taking the absolute value
    sgens = df.clip(upper=0).abs().add_suffix('_sgen').round(5)

    # Combine them into a single dataframe
    processed_df = pd.concat([loads, sgens], axis=1)

    return processed_df


def add_dhn_assets_to_pandapower_excel_sheets(
    all_sheets: dict[str, pd.DataFrame],
    links_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Adds DHN assets to the 'load' or 'sgen' sheets describing the pandapower network.

    This function reads the electricity consumption and generation files from the heat
    results directory to identify which DHN assets act as consumers (loads) and which act
    as generators (sgens) in the electricity network. It then adds these assets to the
    appropriate sheets in the pandapower network excel data.

    Args:
    -----
        all_sheets: Dictionary of DataFrames representing each sheet in the pandapower network excel file.
        links_df: DataFrame containing the PyPSA links.csv data mapping DHN assets to electricity network buses.
    """
    # Get electricity consuming or generating assets based on their carriers
    consumption_assets = []
    generation_assets = []

    for link in links_df['Link']:
        carrier = links_df.loc[links_df['Link'] == link, 'carrier'].values[0]
        if carrier in CARRIERS_ELEC_CONS_LINKS:
            consumption_assets.append(link)
        if carrier in CARRIERS_ELEC_PROD_LINKS:
            generation_assets.append(link)

    if not consumption_assets and not generation_assets:
        logger.warning("No DHN assets found for electricity consumption or generation based on provided carriers.")

    dhn_assets = links_df[
        links_df["electricity_network_bus"].notna() & (links_df["electricity_network_bus"] != 0)
    ].copy()

    bus_sheet_df = all_sheets["bus"]
    elec_network_bus_names_cleaned = bus_sheet_df['name'].astype(str).str.strip()
    dhn_elec_bus_names_cleaned = dhn_assets['electricity_network_bus'].astype(str).str.strip()
    missing_buses = set(dhn_elec_bus_names_cleaned) - set(elec_network_bus_names_cleaned)
    if missing_buses:
        raise ValueError(f"Buses from DHN assets not found in the 'bus' sheet: {missing_buses}")

    bus_name_idx_mapping = dict(zip(elec_network_bus_names_cleaned, bus_sheet_df.index))
    dhn_assets['bus_idx'] = dhn_elec_bus_names_cleaned.map(bus_name_idx_mapping).astype(int)

    # Separate the assets based on their type
    assets_as_loads_info = dhn_assets[dhn_assets['Link'].isin(consumption_assets)]
    assets_as_sgens_info = dhn_assets[dhn_assets['Link'].isin(generation_assets)]

    # Add assets to the 'load' sheet
    if not assets_as_loads_info.empty:
        load_sheet_df = all_sheets["load"]

        # Prepare the new rows to be added as loads.
        new_loads = pd.DataFrame({
            'name': assets_as_loads_info['Link'],
            'bus': assets_as_loads_info['bus_idx'],
            'p_mw': 0.0,
            'q_mvar': 0.0,
            "const_z_percent": 0.0,
            "const_i_percent": 0.0,
            "sn_mva": pd.NA,
            "scaling": 1.0,
            "in_service": True,
            "type": "wye",
        })

        # Determine the starting index for the new loads to ensure it continuous from existing ones.
        last_index = load_sheet_df.index.max() if not load_sheet_df.index.empty else -1
        new_loads.index = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + len(new_loads))
        all_sheets["load"] = pd.concat([load_sheet_df, new_loads])
        logger.info(f"Added {len(new_loads)} DHN assets to the 'load' sheet.")
    else:
        logger.info("No DHN assets to add to the 'load' sheet.")

    # Add assets to the 'sgen' sheet
    if not assets_as_sgens_info.empty:
        sgen_sheet_df = all_sheets["sgen"]

        # Prepare the new rows to be added as sgens.
        new_sgens = pd.DataFrame({
            'name': assets_as_sgens_info['Link'],
            'bus': assets_as_sgens_info['bus_idx'],
            'p_mw': 0.0,
            'q_mvar': 0.0,
            "sn_mva": pd.NA,
            "scaling": 1.0,
            "in_service": True,
            "type": "wye",
        })

        last_index = sgen_sheet_df.index.max() if not sgen_sheet_df.index.empty else -1
        new_sgens.index = pd.RangeIndex(start=last_index + 1, stop=last_index + 1 + len(new_sgens))
        all_sheets["sgen"] = pd.concat([sgen_sheet_df, new_sgens])
        logger.info(f"Added {len(new_sgens)} DHN assets to the 'sgen' sheet.")
    else:
        logger.info("No DHN assets to add to the 'sgen' sheet.")

    return all_sheets


if __name__ == "__main__":
    main()
