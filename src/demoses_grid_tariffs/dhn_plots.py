"""Functions to plot the district heating network (dhn) results."""
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from demoses_grid_tariffs.helper_functions import (
    customize_and_save_plot,
    get_assets_based_on_carrier_name,
    get_electricity_consumption_of_assets,
    get_electricity_generation_of_assets,
)

CARRIERS_GENS_TO_EXCLUDE = {"Electricity", "Geothermal well"}  # Generator carriers to exclude from heat production.

# Set of link carriers to exclude from heat production.
CARRIERS_LINKS_TO_EXCLUDE = {
    "Heat",
    "Heat pipeline",
    "Geothermal ASHP",
    "HT-ATES-charger",
    "HT-ATES-discharger",
    "HT-ATES-discharger-low-heat",
    "HT-ATES-ASHP",
}

SU_CARRIERS = [
    "WT-storage",
]

STORE_CARRIERS = [
    "HT-ATES-store",
]

GEN_CARRIERS = [
    "Residual heat",
    "Solar thermal",
]

link_CARRIERS = [
    "Geothermal",
    "Heat pipeline",
    "Boiler-greengas",
    "Boiler-hydrogen",
    "CHP-greengas",
    "CHP-hydrogen",
    "Waste incineration",
    "Electric boiler",
    "ASHP",
    "HT-ATES-discharger",
]

COMP_TYPE_CARRIERS = {
    "Generator": GEN_CARRIERS,
    "Link": link_CARRIERS,
    "StorageUnit": SU_CARRIERS,
    "Store": STORE_CARRIERS,
}

CARRIERS_NICE_NAMES_AND_COLORS = {
    "Electricity": {"nice_name": "Electricity", "color": "#f94144"},
    "Green gas": {"nice_name": "Green gas", "color": "#f8961e"},
    "Heat": {"nice_name": "Heat", "color": "#4f2e1e"},
    "Heat sink": {"nice_name": "Heat sink", "color": "#523c46"},
    "Low-grade heat": {"nice_name": "Low-grade heat", "color": "#90be6d"},
    "Subsurface heat": {"nice_name": "Subsurface heat", "color": "#277da1"},
    "Geothermal": {"nice_name": "Geothermal", "color": "#0E8CED"},
    "Geothermal well": {"nice_name": "Geothermal well", "color": "#43aa8b"},
    "Geothermal ASHP": {"nice_name": "Geothermal heat pump", "color": "#43aa8b"},
    "Hydrogen": {"nice_name": "Hydrogen", "color": "#577590"},
    "Waste": {"nice_name": "Waste", "color": "#8338ec"},
    "Residual heat": {"nice_name": "Residual heat", "color": "#c0e3f9"},
    "Heat pipeline": {"nice_name":"Heat pipeline", "color": "#87117b"},
    "Boiler-greengas": {"nice_name": "Boiler-greengas", "color": "#0E9B5C"},
    "Boiler-hydrogen": {"nice_name": "Boiler-hydrogen", "color": "#FF8400"},
    "CHP-greengas": {"nice_name": "CHP-greengas", "color": "#04FD00"},
    "CHP-hydrogen": {"nice_name": "CHP-hydrogen", "color": "#B75F07"},
    "Solar thermal": {"nice_name": "Solar thermal", "color": "#f9d32b"},
    "Waste incineration": {"nice_name": "Waste-to-energy", "color": "#FF1493"},
    "Electric boiler": {"nice_name": "Electric boiler", "color": "#2c3fd4"},
    "ASHP": {"nice_name": "Air heat pump", "color": "#9B10E6"},
    "WT-storage": {"nice_name": "Storage-TES", "color": "#6609e8"},
    "HT-ATES-store": {"nice_name": "HT-ATES spillage", "color": "#76c893"},
    "HT-ATES-charger": {"nice_name": "HT-ATES charging", "color": "#8c350f"},
    "HT-ATES-discharger": {"nice_name": "Storage-HT-ATES", "color": "#F50E0E"},
    "HT-ATES-ASHP": {"nice_name": "HT-ATES heat pump", "color": "#f9844a"},
    "HT-ATES-discharger-low-heat": {"nice_name": "HT-ATES-discharger-low-heat", "color": "#4d908e"},
}

CARRIERS_RENAME_MAP = {k: v["nice_name"] for k, v in CARRIERS_NICE_NAMES_AND_COLORS.items()}


def plot_dhn_results(n: pypsa.Network, output: Path) -> None:
    """Function to plot the results of the optimized district heating network.

    Args:
    -----
        n: PyPSA network object.
        output: Path where the figures will be saved.
    """
    plot_heat_technology_mix_for_given_network(network=n, output_dir=output)
    plot_combined_heat_analysis(n, output=output)
    plot_heat_dispatch_timeseries(n, output=output)
    plot_electricity_consumption_and_generation(n, output=output)
    plot_optimal_capacities_per_tech_category(n, color="#87117b", output=output)
    plot_heat_load_duration_curve(n, output=output)  # Already part of the combined heat analysis plot.
    plot_heat_production_per_tech_category(n, output=output)  # Already part of the combined heat analysis plot.

    #  Plot state of charge / energy level of storage units and stores
    for comp_type in ["StorageUnit", "Store"]:
        units = get_assets_based_on_carrier_name(
            n,
            component_type=comp_type,
            carrier_name=SU_CARRIERS[0] if comp_type == "StorageUnit" else STORE_CARRIERS[0],
        )
        plot_soc_energy_level(
            n,
            units=units,
            component_type=comp_type,
            output=output,
        )

    # Plot low-level results in a subfolder
    output.mkdir(parents=True, exist_ok=True)
    detail_output = output / "detailed_plots"
    detail_output.mkdir(parents=True, exist_ok=True)

    # Plot heat production per technology units
    plot_heat_production_per_tech_units(n, output=detail_output)

    # Plot optimal capacities per technology units
    for component_type, carriers in COMP_TYPE_CARRIERS.items():
        for carrier_name in carriers:
            color = CARRIERS_NICE_NAMES_AND_COLORS.get(carrier_name, {}).get('color', "#EF5D53")
            plot_optimal_capacities_per_tech_units(
                n,
                component_type=component_type,
                carrier_name=carrier_name,
                color=color,
                output=detail_output,
            )


def plot_heat_technology_mix_for_given_network(
    network: pypsa.Network,
    name: str = "",
    file_extension: str = "pdf",
    output_dir: Path = None,
    font_size: int = 10,
) -> None:
    """Plot the heat technology mix for a given network in least-cost scenario."""
    # keep "Heat pipeline" and "HT-ATES-discharger" in the analysis
    carrier_link_to_exclude = CARRIERS_LINKS_TO_EXCLUDE.copy()
    for carrier in ["Heat pipeline", "HT-ATES-discharger"]:
        carrier_link_to_exclude.remove(carrier)

    CARRIERS_TO_EXCLUDE = CARRIERS_GENS_TO_EXCLUDE.union(carrier_link_to_exclude)
    # Extract optimized capacities per technology
    gens_df = network.generators[
        ~network.generators.carrier.isin(CARRIERS_TO_EXCLUDE)
    ].groupby("carrier")["p_nom_opt"].sum()

    links_df = network.links[
        ~network.links.carrier.isin(CARRIERS_TO_EXCLUDE)
    ].groupby("carrier")["p_nom_opt"].sum()

    storage_df = network.storage_units.groupby("carrier")["p_nom_opt"].sum()

    techs_cap_df = pd.concat([gens_df, links_df, storage_df], axis=0).fillna(0)
    techs_cap_df.rename(index=CARRIERS_RENAME_MAP, inplace=True)

    storage_techs = [CARRIERS_RENAME_MAP["WT-storage"], CARRIERS_RENAME_MAP["HT-ATES-discharger"]]
    pipeline_tech = CARRIERS_RENAME_MAP["Heat pipeline"]

    # Separate the data for plotting
    supplying_s = techs_cap_df.drop(storage_techs + [pipeline_tech], errors="ignore").sort_index()
    storage_s = techs_cap_df[techs_cap_df.index.isin(storage_techs)].sort_index()
    pipeline_s = techs_cap_df[techs_cap_df.index == pipeline_tech]

    # Get colors for carriers
    carr_colors = {info["nice_name"]: info["color"] for info in CARRIERS_NICE_NAMES_AND_COLORS.values()}
    # For Geothermal heat pump and HT-ATES heat pump, use same color as Geothermal and HT-ATES discharger
    carr_colors[CARRIERS_RENAME_MAP["Geothermal ASHP"]] = carr_colors[CARRIERS_RENAME_MAP["Geothermal"]]
    carr_colors[CARRIERS_RENAME_MAP["HT-ATES-ASHP"]] = carr_colors[CARRIERS_RENAME_MAP["HT-ATES-discharger"]]

    # Plotting
    fig = plt.figure(figsize=(8, 5))
    outer_grid = GridSpec(1, 1, figure=fig)
    main_ax = fig.add_subplot(outer_grid[0])
    # main_ax.set_title(name, fontsize=font_size + 4, pad=18)
    main_ax.axis("off")

    inner_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0], wspace=0.45)
    ax1 = fig.add_subplot(inner_grid[0])
    ax2 = fig.add_subplot(inner_grid[1])
    ax3 = fig.add_subplot(inner_grid[2])

    y_ticks_fontsize = font_size + 1
    bottom = 0
    bar_width = 0.15
    bar_x, bar_y = -0.10, 0.12

    # Panel 1: primary supply capacity
    for tech, value in supplying_s.items():
        ax1.bar(
            "Primary Supply",
            value,
            width=bar_width,
            bottom=bottom,
            label=tech,
            alpha=1,
            color=carr_colors.get(tech, "#000000"),
        )
        bottom += value

    ax1.set_ylim(0, 1400)
    ax1.tick_params(axis="y", labelsize=y_ticks_fontsize)
    ax1.set_xlim(bar_x, bar_y)

    # Panel 2: storage discharge capacity
    bottom = 0
    for tech, value in storage_s.items():
        ax2.bar(
            "Storage",
            value,
            width=bar_width,
            bottom=bottom,
            label=tech,
            alpha=0.65,
            edgecolor="white",
            hatch="/",
            color=carr_colors.get(tech, "#000000"),
        )
        bottom += value

    ax2.set_ylim(0, 500)
    ax2.tick_params(axis="y", labelsize=y_ticks_fontsize)
    ax2.set_xlim(bar_x, bar_y)

    # Panel 3: pipeline capacity
    if not pipeline_s.empty:
        ax3.bar(
            pipeline_s.index,
            pipeline_s.values,
            width=bar_width,
            label=pipeline_s.index[0],
            alpha=0.7,
            edgecolor="white",
            hatch="/",
            color=carr_colors.get(pipeline_s.index[0], "#000000"),
        )

    ax3.set_ylim(0, 800)
    ax3.tick_params(axis="y", labelsize=y_ticks_fontsize)
    ax3.set_xlim(bar_x, bar_y)


    # Labels and aesthetics
    ax1.set_ylabel("Deployed capacity  [MW]", fontsize=y_ticks_fontsize + 1, labelpad=12)

    for ax, label in zip(
        [ax1, ax2, ax3],
        ["Primary supply \ncapacity", "Storage discharge \ncapacity", "Heat pipeline \ncapacity"],
    ):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0])
        ax.set_xticklabels([label], rotation=0, ha="center", fontsize=y_ticks_fontsize - 1)

        # Increase the 'pad' argument to move labels away from the axis
        ax.tick_params(axis='x', pad=15)

    # Merged legend
    all_handles, all_labels = [], []

    for ax in [ax1, ax2, ax3]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_labels.append(label)
                all_handles.append(handle)

    fig.legend(
        all_handles,
        all_labels,
        ncol=1,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.88, 0.4),
        fontsize=font_size - 1,
    )

    fig.subplots_adjust(left=0.12, bottom=0.2, right=0.78, top=0.96)

    # Save
    filename = f"technology_mix_{name}.{file_extension}"
    if output_dir:
        plt.savefig(output_dir / filename, dpi=300)
    else:
        plt.savefig(filename, dpi=300)

    plt.close(fig)


def plot_optimal_capacities_per_tech_category(
    network: pypsa.Network, color: str = "#EF5D53", output: Path = None,
) -> None:
    """Plot the optimal and potential capacities of technology categories aggregated by carrier.

    Args:
    -----
        network: PyPSA network object.
        output: Path where the figures will be saved.
        color: Color for the bars in the plot.
    """
    # Helper function to aggregate data
    def aggregate_capacity(df: pd.DataFrame, carriers: list) -> pd.DataFrame:
        return (
            df[df.carrier.isin(carriers)]
            .groupby("carrier")[["p_nom_opt", "p_nom_max"]]
            .sum()
        )

    # Generators
    gen_carriers = ["Residual heat", "Solar thermal"]
    gens_df = aggregate_capacity(network.generators, gen_carriers)

    # Links
    links_to_exclude = [
        "Heat",
        "Geothermal ASHP",
        "HT-ATES-ASHP",
        "HT-ATES-charger",
        "HT-ATES-discharger-low-heat",
    ]
    links_df = (
        network.links[~network.links.carrier.isin(links_to_exclude)]
        .groupby("carrier")[["p_nom_opt", "p_nom_max"]]
        .sum()
    )

    # Storage units
    storage_df = network.storage_units.groupby("carrier")[["p_nom_opt", "p_nom_max"]].sum()

    # Combine all technology capacities into a single DataFrame and separate the data for plotting
    all_techs_df = pd.concat([gens_df, links_df, storage_df], axis=0).fillna(0)

    # Rename carriers for better readability
    all_techs_df.rename(index=CARRIERS_RENAME_MAP, inplace=True)

    # Separate the data for plotting using the NEW names
    heat_pipeline_df = all_techs_df[all_techs_df.index == CARRIERS_RENAME_MAP["Heat pipeline"]]
    other_techs_df = all_techs_df[all_techs_df.index != CARRIERS_RENAME_MAP["Heat pipeline"]].sort_index()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [12, 1]})

    # Plot other_techs_df (ax1)
    # Plot p_nom_max as background bars
    ax1.bar(
        other_techs_df.index,
        other_techs_df['p_nom_max'],
        color=color,
        alpha=0.3,
        edgecolor="white",
        width=0.9,
        label="Potential capacity",
    )
    # Plot p_nom_opt as foreground bars
    ax1.bar(
        other_techs_df.index,
        other_techs_df['p_nom_opt'],
        color=color,
        alpha=0.9,
        edgecolor="white",
        hatch="/",
        width=0.9,
        label="Cost-optimized capacity",
    )

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel("Capacity [MW]", fontsize=11)
    ymax1 = other_techs_df['p_nom_max'].max() * 1.05
    ax1.set_ylim(0, ymax1)
    ax1.tick_params(axis='x', rotation=90)

    # Plot heat_pipeline_df (ax2)
    ax2.bar(
        heat_pipeline_df.index,
        heat_pipeline_df['p_nom_max'],
        color=color,
        alpha=0.3,
        edgecolor="white",
        width=0.5,
    )
    ax2.bar(
        heat_pipeline_df.index,
        heat_pipeline_df['p_nom_opt'],
        color=color,
        alpha=0.9,
        edgecolor="white",
        hatch="/",
        width=0.5,
    )

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ymax2 = heat_pipeline_df['p_nom_max'].max() * 1.05
    ax2.set_ylim(0, ymax2)
    ax2.tick_params(axis='x', rotation=90)


    # Add a common legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,  # Align horizontally
        frameon=False,
        fontsize='medium'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output:
        plt.savefig(output / "optimal_capacities.png",  dpi=300)
    else:
        plt.savefig("optimal_capacities.png",  dpi=300)
    plt.close(fig)


def plot_optimal_capacities_per_tech_units(
    n: pypsa.Network,
    component_type: str,
    carrier_name: str,
    color: str = "#8AEF53",
    output: Path = None,
) -> None:
    """Plots the optimal and potential capacity for each unit of a given carrier.

    Args:
    -----
        n: PyPSA network object.
        component_type: 'generator', 'link', or 'storage_unit'.
        carrier_name: Name of the carrier to filter units.
        color: Color for the bars in the plot.
        output: Path where the figure will be saved. If None, saves in the current directory
    """
    if component_type.lower() not in ["generator", "link", "storageunit", "store"]:
        raise ValueError(f"Invalid {component_type=}. Choose 'generator', 'link', 'storageunit', or 'store'.")

    # Get the appropriate DataFrame based on component type
    df_name = f"{component_type.lower()}s" if component_type.lower() != "storageunit" else "storage_units"
    component_df = getattr(n, df_name)

    # Filter the DataFrame for the specified carrier
    carrier_df = component_df[component_df.carrier == carrier_name].copy()

    if carrier_df.empty:
       raise ValueError(f"No units found for carrier '{carrier_name}' in component type '{component_type}'.")

    # Prepare data for plotting, replacing infinite p_nom_max with NaN
    max_cap_name = 'p_nom_max' if 'p_nom_max' in carrier_df.columns else 'e_nom_max'
    carrier_df['max_cap'] = carrier_df[max_cap_name].replace(np.inf, np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot p_nom_max as background bars (if it has finite values)
    if carrier_df['max_cap'].notna().any():
        ax.bar(
            carrier_df.index,
            carrier_df['max_cap'],
            color=color,
            alpha=0.3,
            edgecolor="white",
            width=0.8,
            label="Potential capacity",
        )

    # Plot opt_cap_name as foreground bars
    opt_cap_name = 'p_nom_opt' if 'p_nom_opt' in carrier_df.columns else 'e_nom_opt'
    ax.bar(
        carrier_df.index,
        carrier_df[opt_cap_name],
        color=color,
        alpha=0.9,
        edgecolor="white",
        hatch="/",
        width=0.8,
        label="Cost-optimized capacity",
    )


    # Set ylabel
    y_label = "Capacity [MW]" if component_type.lower() in ["generator", "link"] else "Capacity [MWh]"
    ax.set_ylabel(y_label, fontsize=10)

    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limits (use the maximum of opt_cap_name and the finite max_cap for the limit)
    ymax = max(carrier_df[opt_cap_name].max(), carrier_df['max_cap'].max()) * 1.05
    # Handle case where all capacities are zero
    if ymax == 0:
        ymax = 1
    ax.set_ylim(0, ymax)

    plt.xticks(rotation=90)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
        fontsize='medium'
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    filename = f"{carrier_name.replace(' ', '_').lower()}_units_capacity.png"
    if output:
        output.mkdir(parents=True, exist_ok=True)
        plt.savefig(output / filename, dpi=300)
    else:
        plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_heat_production_per_tech_category(
    n: pypsa.Network, output: Path, figsize: tuple = (6, 4), threshold: float = 0.5,
) -> None:
    """Generates a pie chart of the total heat production share per carrier.

    Args:
    -----
        n: PyPSA network object.
        figsize: Size of the figure.
        threshold: Minimum percentage threshold to display autopct labels (wedges below this will not show labels).
        output: Path where the figure will be saved.
    """
    # Calculate heat production from generators and links, excluding specified carriers
    gens_to_include = n.generators[~n.generators.carrier.isin(CARRIERS_GENS_TO_EXCLUDE)]
    gen_production = n.generators_t.p[gens_to_include.index].sum().groupby(gens_to_include.carrier).sum()
    links_to_include = n.links[~n.links.carrier.isin(CARRIERS_LINKS_TO_EXCLUDE)]
    link_production = (-1 * n.links_t.p1[links_to_include.index]).sum().groupby(links_to_include.carrier).sum()
    calculated_production = gen_production.add(link_production, fill_value=0)

    # Define the set of primary heat producing carriers
    gen_carriers = n.generators[~n.generators.carrier.isin(CARRIERS_GENS_TO_EXCLUDE)].carrier.unique()
    link_carriers = n.links[~n.links.carrier.isin(CARRIERS_LINKS_TO_EXCLUDE)].carrier.unique()
    primary_heat_carriers = sorted([CAR for CAR in set(gen_carriers).union(set(link_carriers))])

    # Align calculated production with the complete list of primary heat carriers
    total_production = calculated_production.reindex(primary_heat_carriers, fill_value=0)
    plot_data = total_production[total_production > 0.01].sort_values(ascending=False)

    # Create color and name maps
    nice_name_map = {k: v["nice_name"] for k, v in CARRIERS_NICE_NAMES_AND_COLORS.items()}
    color_map = {k: v["color"] for k, v in CARRIERS_NICE_NAMES_AND_COLORS.items()}
    plot_data_nice_names = plot_data.rename(index=nice_name_map)

    def conditional_autopct(pct: float, threshold: float) -> str:
        return f'{pct:.1f}%' if pct >= threshold else ''

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    wedges, texts, autotexts = ax.pie(
        plot_data_nice_names,
        autopct=lambda pct: conditional_autopct(pct, threshold=threshold),
        startangle=90,
        colors=[color_map[carrier] for carrier in plot_data.index],
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        labels=None
    )
    plt.setp(autotexts, size=7, weight="bold", color="white")

    # Create custom legend
    legend_handles = []
    for carrier in primary_heat_carriers:
        patch = mpatches.Patch(color=color_map[carrier], label=nice_name_map[carrier])
        legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.9, 0.8),
        fontsize=9,
        frameon=False,
    )

    # Finalize and save the plot
    total_gwh = total_production.sum() / 1000
    ax.set_title(
        f"Total heat production: {total_gwh:.0f} GWh/year",
        fontsize=10,
        pad=12,
    )
    ax.axis('equal')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output / "heat_production_share.png", dpi=300)
    plt.close(fig)


def plot_heat_production_per_tech_units(n: pypsa.Network, output: Path) -> None:
    """Plot the total heat production per heat producing asset.

    Args:
    -----
        n: PyPSA network object.
        output: Path where the figure will be saved.
    """
    # Compute total annual energy production per heat producing asset
    asset_heat_production = {}

    # Assets modeled as generators
    carriers_all_gens = set(n.generators.carrier)
    carriers_gens_to_include = carriers_all_gens.difference(CARRIERS_GENS_TO_EXCLUDE)
    gens_to_include = []
    for carrier in carriers_gens_to_include:
        gens_to_include.extend(get_assets_based_on_carrier_name(n, component_type="Generator", carrier_name=carrier))

    for gen in gens_to_include:
        asset_heat_production[gen] = n.generators_t.p[gen].sum() / 1000  # Convert to GWh

    # Assets modeled as links (assuming their heat production is through p1 especially for multi-output links like CHPs)
    carriers_all_links = set(n.links.carrier)
    carriers_links_to_include = carriers_all_links.difference(CARRIERS_LINKS_TO_EXCLUDE)
    links_to_include = []
    for carrier in carriers_links_to_include:
        links_to_include.extend(get_assets_based_on_carrier_name(n, component_type="Link", carrier_name=carrier))

    for link in links_to_include:
        asset_heat_production[link] = n.links_t.p1[link].sum() * -1 / 1000  # Convert to GWh

    # Sort by production values for better visualization
    asset_heat_production = dict(sorted(asset_heat_production.items(), key=lambda item: item[1], reverse=True))

    # Plot bar chart
    plt.figure(figsize=(16, 12), dpi=150)
    plt.bar(asset_heat_production.keys(), asset_heat_production.values(), color="#EF5D53", edgecolor="white")
    plt.ylabel("Heat production [GWh]", fontsize=14, labelpad=10)
    plt.yticks(fontsize=12)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--")
    plt.savefig(output / "assets_heat_production.png")
    plt.close()


def plot_electricity_consumption_and_generation(n: pypsa.Network, output: Path) -> None:
    """Plot the total electricity consumption and generation of the district heating network.

    Args:
    -----
        n: PyPSA network object.
        output: Path where the figure will be saved.
    """
    elec_consumption_series = get_electricity_consumption_of_assets(n).sum(axis=1)
    elec_generation_series = get_electricity_generation_of_assets(n).sum(axis=1)
    plot_df = pd.DataFrame({"Consumption": elec_consumption_series, "Generation": elec_generation_series})

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    plot_df.plot(ax=ax, color=["#EF5D53", "#00ffc3"])

    customize_and_save_plot(
        ax=ax,
        output_dir=output,
        filename="electric_power.png",
        ylabel="Electric power [MW]",
        title="",
        fontsize=13,
        add_grid=False,
        add_legend=True,
        legend_handles=ax.get_lines(),
        legend_labels=plot_df.columns.tolist(),
    )


def plot_heat_dispatch_timeseries(n: pypsa.Network, output: Path) -> None:
    """Plot the dispatch of the district heating network.

    Args:
    -----
        n: PyPSA network object.
        output: Path where the figure will be saved.
    """
    # Add nice names and colors to carriers
    for carrier in n.carriers.index:
        if carrier in CARRIERS_NICE_NAMES_AND_COLORS:
            n.carriers.at[carrier, 'nice_name'] = CARRIERS_NICE_NAMES_AND_COLORS[carrier]['nice_name']
            n.carriers.at[carrier, 'color'] = CARRIERS_NICE_NAMES_AND_COLORS[carrier]['color']

    # Plot heat dispatch area plot
    fig, ax, g = n.statistics.energy_balance.plot.area(linewidth=0, bus_carrier="Heat", figsize=(7, 4))

    # # Plot load
    # load = -1 * n1.loads_t.p_set.sum(axis=1)
    # load.plot(ax=ax, color="#4f2e1e", linewidth=0.5, label='Heat load', kind='area', alpha=0.7)

    ax.set_ylabel("Heat dispatch  [MW]")
    ax.grid(True, axis='y', alpha=0.7, linestyle='--', linewidth=0.2)

    # Update legend to remove the 'Heat' entry
    handles, labels = ax.get_legend_handles_labels()
    unique_legend_items = dict(zip(labels, handles))  # Use a dictionary to get unique handles and labels.

    label_to_remove = ["Heat", "Heat sink"]
    for label in label_to_remove:
        unique_legend_items.pop(label, None)

    # Convert the unique dictionary back into separate lists for the legend function
    final_labels = list(unique_legend_items.keys())
    final_handles = list(unique_legend_items.values())

    # First remove any existing legends to ensure a clean slate before adding the new one
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if g._legend is not None:
        g._legend.remove()

    # Add the updated legend
    ax.legend(final_handles, final_labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', frameon=False)

    fig.tight_layout()
    plt.savefig(output / "dispatch_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_soc_energy_level(
    n: pypsa.Network,
    units: list,
    component_type: str,
    color: str = "#EF5D53",
    fontsize: float = 11,
    capacity_threshold: float = 1.0,
    output: Path = None,
) -> None:
    """Plots the state of charge (SOC) or energy level of storage or store units.

    Args:
    -----
        n: PyPSA network object.
        units: List of store or storage unit names to plot.
        component_type: 'store' or 'storage_unit'.
        color: Color for the plot lines.
        fontsize: Font size for titles and labels.
        capacity_threshold: Minimum capacity (in MW or MWh) to include a unit in the plot.
        output: Path where the figure will be saved. If None, saves in the current directory.
    """
    if component_type.lower() not in ['store', 'storageunit']:
        raise ValueError(f"Invalid {component_type=}. Choose 'store' or 'storageunit'.")

    if component_type.lower() == 'store':
        timeseries_df = n.stores_t.e
        static_df = n.stores
        capacity_col = 'e_nom_opt'
        ylabel = "Energy [MWh]"
    else:
        timeseries_df = n.storage_units_t.state_of_charge
        static_df = n.storage_units
        capacity_col = 'p_nom_opt'
        ylabel = "Energy [MWh]"

    # Check if the units are in the network
    missing_units = [u for u in units if u not in static_df.index]
    if missing_units:
        raise ValueError(f"The following units are not found in the network {component_type}: {missing_units}")

    # Pre-filter the list to include only significant units
    significant_units = [u for u in units if static_df.at[u, capacity_col] > capacity_threshold]

    # Proceed only if there are significant units to plot
    num_units = len(significant_units)
    if num_units == 0:
        print(f"Warning: No units met the {capacity_threshold=} MW/MWh for plotting '{component_type}'.")
        return

    data_to_plot = timeseries_df[significant_units]

    # Set up the plot grid based on the number of significant units
    ncols = 2
    nrows = math.ceil(num_units / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5), sharex=True)

    # Handle case where subplots returns a single ax object
    axes = np.atleast_1d(axes).flatten()

    # Loop and plot only the significant units
    for i, unit_name in enumerate(data_to_plot.columns):
        ax = axes[i]
        unit_data = data_to_plot[unit_name]

        ax.set_title(unit_name, fontsize=fontsize + 1)
        unit_data.plot(ax=ax, color=color, linewidth=2)

        # To hide noise near zero, clip y-axis only if data is large enough
        if unit_data.max() > 1:
            ax.set_ylim(bottom=1)

        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=12)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.margins(x=0.01)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Set x-labels for the bottom row of subplots
    start_index = max(0, num_units - ncols)
    for i in range(start_index, num_units):
        axes[i].tick_params(axis='x', labelbottom=True, labelsize=fontsize)
        axes[i].set_xlabel("snapshot", fontsize=fontsize+1)

    # Hide any unused subplots
    for j in range(num_units, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 1])
    filename = f"{component_type}_soc.png"
    if output:
        output.mkdir(parents=True, exist_ok=True)
        plt.savefig(output / filename, dpi=300)
    else:
        plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_heat_load_duration_curve(n: pypsa.Network, output: Path) -> None:
    """Plot the heat load duration curve of the district heating network.

    Args:
    -----
        n: PyPSA network object.
        output: Path where the figure will be saved.
    """
    # Extract heat dispatch from generators and links, excluding specified carriers
    gens_to_include = n.generators[~n.generators.carrier.isin(CARRIERS_GENS_TO_EXCLUDE)]
    gen_dispatch = n.generators_t.p[gens_to_include.index]
    links_to_include = n.links[~n.links.carrier.isin(CARRIERS_LINKS_TO_EXCLUDE)]
    link_dispatch = -1 * n.links_t.p1[links_to_include.index]

    # Combine and group dispatch by carrier
    total_dispatch_components = pd.concat([gen_dispatch, link_dispatch], axis=1)
    carrier_map = pd.concat([n.generators.carrier, n.links.carrier])
    dispatch_by_carrier = total_dispatch_components.T.groupby(carrier_map).sum().T
    dispatch_by_carrier[dispatch_by_carrier < 0] = 0

    # Calculate total load and sort the data chronologically by load
    total_hourly_load = dispatch_by_carrier.sum(axis=1)
    sorted_index = total_hourly_load.sort_values(ascending=False).index
    sorted_dispatch = dispatch_by_carrier.loc[sorted_index]

    # Calculate total MWh produced by each carrier over the year
    total_energy_produced = dispatch_by_carrier.sum()

    # Sort carriers from most production (baseload) to least (peaker)
    merit_order = total_energy_produced.sort_values(ascending=False).index

    # Re-order the columns of the DataFrame according to the merit order
    sorted_dispatch = sorted_dispatch[merit_order]

    # Reset index and plot the sorted data
    sorted_dispatch.reset_index(drop=True, inplace=True)

    # Re-order the DataFrame columns based on merit order first
    sorted_dispatch = sorted_dispatch[merit_order]

    # Then move specific carriers to the bottom of the stack (are usually baseloads)
    bottom_carriers = ["Geothermal", "Residual heat", "Solar thermal", "ASHP"]
    other_carriers = [col for col in sorted_dispatch.columns if col not in bottom_carriers]
    final_column_order = [carr for carr in bottom_carriers if carr in sorted_dispatch.columns] + other_carriers

    sorted_dispatch = sorted_dispatch[final_column_order]

    sorted_dispatch.reset_index(drop=True, inplace=True)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    color_map = {
        carrier: info["color"]
        for carrier, info in CARRIERS_NICE_NAMES_AND_COLORS.items()
    }
    nice_name_map = {
        carrier: info["nice_name"]
        for carrier, info in CARRIERS_NICE_NAMES_AND_COLORS.items()
    }

    plot_df = sorted_dispatch.rename(columns=nice_name_map)
    plot_colors = [color_map.get(col) for col in sorted_dispatch.columns]

    plot_df.plot.area(
        ax=ax,
        stacked=True,
        linewidth=0,
        color=plot_colors
    )

    # Customize and save the plot
    ax.set_xlabel("Duration [hours]")
    ax.set_ylabel("Heat dispatch [MW]")
    ax.grid(True, axis='y', alpha=0.7, linestyle='--', linewidth=0.2)
    ax.set_xlim(0, len(sorted_dispatch))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = []
    for carrier in final_column_order:
        if carrier in nice_name_map:
            patch = mpatches.Patch(color=color_map[carrier], label=nice_name_map[carrier])
            legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1, 0.9),
        loc='upper left',
        fontsize='medium',
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.99, 1])
    output.mkdir(parents=True, exist_ok=True)
    plt.savefig(output / "dispatch_duration_curve.png", dpi=300)
    plt.close(fig)


def plot_combined_heat_analysis(n: pypsa.Network, output: Path, threshold: float = 0.5) -> None:
    """Plot a combined figure with heat load duration curve and heat production pie chart.

    Args:
    -----
        n: PyPSA network object.
        threshold: Minimum percentage threshold to display autopct labels in pie chart.
        output: Path where the figure will be saved.
    """
    gens_to_include = n.generators[~n.generators.carrier.isin(CARRIERS_GENS_TO_EXCLUDE)]
    gen_dispatch = n.generators_t.p[gens_to_include.index]

    links_to_include = n.links[~n.links.carrier.isin(CARRIERS_LINKS_TO_EXCLUDE)]
    link_dispatch = -1 * n.links_t.p1[links_to_include.index]

    total_dispatch_components = pd.concat([gen_dispatch, link_dispatch], axis=1)

    carrier_map = pd.concat([gens_to_include.carrier, links_to_include.carrier])
    dispatch_by_carrier = total_dispatch_components.T.groupby(carrier_map).sum().T
    dispatch_by_carrier[dispatch_by_carrier < 0] = 0

    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        dpi=300,
        gridspec_kw={'width_ratios': [0.62, 0.38]}
    )

    # Plot 1: heat load duration curve (on ax1)
    total_hourly_load = dispatch_by_carrier.sum(axis=1)
    sorted_index = total_hourly_load.sort_values(ascending=False).index
    sorted_dispatch = dispatch_by_carrier.loc[sorted_index]

    total_energy_produced = dispatch_by_carrier.sum()
    merit_order = total_energy_produced.sort_values(ascending=False).index

    # Re-order the DataFrame columns based on merit order first
    sorted_dispatch = sorted_dispatch[merit_order]

    # Then move specific carriers to the bottom of the stack (are usually baseloads)
    bottom_carriers = ["Geothermal", "Residual heat", "Solar thermal", "ASHP"]
    other_carriers = [col for col in sorted_dispatch.columns if col not in bottom_carriers]
    final_column_order = [carr for carr in bottom_carriers if carr in sorted_dispatch.columns] + other_carriers

    sorted_dispatch = sorted_dispatch[final_column_order]

    sorted_dispatch.reset_index(drop=True, inplace=True)

    color_map = {
        carrier: info["color"]
        for carrier, info in CARRIERS_NICE_NAMES_AND_COLORS.items()
    }
    nice_name_map = {
        carrier: info["nice_name"]
        for carrier, info in CARRIERS_NICE_NAMES_AND_COLORS.items()
    }

    plot_df = sorted_dispatch.rename(columns=nice_name_map)
    plot_colors = [color_map.get(col) for col in sorted_dispatch.columns]

    plot_df.plot.area(
        ax=ax1,
        stacked=True,
        linewidth=0,
        color=plot_colors,
        legend=False  # Disable legend for this subplot
    )

    ax1.set_xlabel("Duration [hours]", fontsize=13, labelpad=13)
    ax1.set_ylabel("Heat dispatch [MW]", fontsize=14, labelpad=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.grid(True, axis='y', alpha=0.8, linestyle='--', linewidth=0.2)
    ax1.set_xlim(0, len(sorted_dispatch))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_title("Heat Dispatch Duration Curve", fontsize=12)

    # Plot 2: heat production share pie chart (on ax2)
    gen_production = n.generators_t.p[gens_to_include.index].sum().groupby(gens_to_include.carrier).sum()
    link_production = (-1 * n.links_t.p1[links_to_include.index]).sum().groupby(links_to_include.carrier).sum()
    calculated_production = gen_production.add(link_production, fill_value=0)

    gen_carriers = n.generators[~n.generators.carrier.isin(CARRIERS_GENS_TO_EXCLUDE)].carrier.unique()
    link_carriers = n.links[~n.links.carrier.isin(CARRIERS_LINKS_TO_EXCLUDE)].carrier.unique()
    primary_heat_carriers = sorted(list(set(gen_carriers).union(set(link_carriers))))

    total_production = calculated_production.reindex(primary_heat_carriers, fill_value=0)
    plot_data = total_production[total_production > 0.01].sort_values(ascending=False)
    plot_data_nice_names = plot_data.rename(index=nice_name_map)

    def conditional_autopct(pct: float, threshold: float) -> str:
        return f'{pct:.1f}%' if pct >= threshold else ''

    wedges, texts, autotexts = ax2.pie(
        plot_data_nice_names,
        autopct=lambda pct: conditional_autopct(pct, threshold=threshold),
        startangle=90,
        colors=[color_map.get(carrier) for carrier in plot_data.index],
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        labels=None
    )
    plt.setp(autotexts, size=8, weight="bold", color="white")
    total_gwh = total_production.sum() / 1000
    ax2.set_title(f"Total heat production: {total_gwh:.0f} GWh/year", y=-0.025, fontsize=14, pad=5)

    # Create a common legend for both plots (use the order from the load duration curve for a consistent legend)
    legend_handles = []
    for carrier in final_column_order:
        if carrier in nice_name_map:
            patch = mpatches.Patch(color=color_map[carrier], label=nice_name_map[carrier])
            legend_handles.append(patch)

    fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.56, 0.88), # Position next to the plots
        loc='upper center',
        ncol=1, # Arrange horizontally
        fontsize=12,
        frameon=False
    )

    # Finalize and save plot
    fig.tight_layout(rect=[0, 0, 1, 1])
    output.mkdir(parents=True, exist_ok=True)
    plt.savefig(output / "combined_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
