import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from demoses_grid_tariffs.helper_functions import (
    calculate_violation_metrics,
    customize_and_save_plot,
    get_custom_network_coordinates,
    prepare_graph_from_excel,
    sort_result_df_columns_by_voltage_levels,
)

MIN_VOLTAGE = 0.7   # Min voltage for clipping bus voltage data to avoid extreme outliers
MAX_VOLTAGE = 1.1   # Max voltage for clipping bus voltage data to avoid extreme outliers
MAX_OVERLOAD = 400  # Max overload percentage for clipping line and transformer loading data to avoid extreme outliers

TRAFO_NAMES = {
    "T_Waterningen": "380/150kV Waterningen",
    "T_Krimpen": "380/150kV Krimpen",
    "T_Rijswijk": "150/25kV Rijswijk",
    "T_sGravenhage": "150/25kV s'Gravenhage",
    "T_Voorburg": "150/25kV Voorburg",
    "T_Ommoord_HV": "150/25kV Ommoord",
    "T_Zuidwijk": "150/23kV Zuidwijk",
    "T_Laagveen": "25/10kV Laagveen",
    "T_HVS_Centrale": "25/10kV HVS Centrale",
    "T_Nootdorp2_MV1": "25/23kV Nootdorp",
    "T_Nootdorp2_MV2": "23/10kV Nootdorp",
    "T_Noordsingel": "25/10kV Noordsingel",
    "T_Ommoord_MV": "25/10kV Ommoord",
    "T_Bleiswijk1": "25/10kV Bleiswijk",
}

LINE_NAMES = {
    "150kV_Waterningen_Rijskwijk": "150kV Waterningen--Rijswijk",
    "150kV_sGravenhage_Rijskwijk": "150kV Rijswijk--sGravenhage",
    "150kV_sGravenhage_Voorburg": "150kV sGravenhage--Voorburg",
    "150kV_Voorburg_Waterningen": "150kV Waterningen--Voorburg",
    "150kV_Krimpen_Ommoord(1)": "150kV Krimpen--Ommoord_1",
    "150kV_Krimpen_Ommoord(2)": "150kV Krimpen--Ommoord_2",
    "150kV_Krimpen_Zuidwijk": "150kV Krimpen--Zuidwijk",
    "25kV_HVS_Zuid_Laagveen(1)": "25kV HVS Zuid--Laagveen_1",
    "25kV_HVS_Zuid_Laagveen(2&3)": "25kV HVS Zuid--Laagveen_2&3",
    "25kV_HVS_Zuid_Laagveen(4)": "25kV HVS Zuid--Laagveen_4",
    "25kV_HVS_Centrale_DH_EB": "25kV HVS Centrale--DH_1",
    "25kV_HVS_Centrale_DH_HP": "25kV HVS Centrale--DH_2",
    "25kV_DH_CHP_Cables": "25kV DH--HVS Centrale--DH_3",
    "25kV_HVS_Oost_Nootdorp2": "25kV HVS Oost--Nootdorp",
    "25kV_HVS_Oost_Noordsingel(1)": "25kV HVS Oost--Noordsingel_1",
    "25kV_HVS_Oost_Noordsingel(2)": "25kV HVS Oost--Noordsingel_2",
    "25kV_Ommoord_RoCa(1&2)": "25kV RoCa--Ommoord",
    "25kV_Ommoord_Bleiswijk(1&2)": "25kV Ommoord--Bleiswijk",
    "23kV_Zuidwijk_RDAM": "23kV Zuidwijk--RDAM",
    "10kV_Laagveen_DH_1": "10kV Laagveen--DH_1",
    "10kV_HVS_Centrale_DH_2": "10kV HVS Centrale--DH_2",
    "10kV_Nootdorp2_DH_3": "10kV Nootdorp--DH_3",
    "10kV_Noordsingel_DH_4": "10kV Noordsingel--DH_4",
    "10kV_Ommoord_RDAM": "10kV Ommoord--RDAM",
    "10kV_Bleiswijk_Tuinders": "10kV Bleiswijk--Tuinders",
}

BUS_NAMES = {
    # 380 kV Level
    "b_380_grid": "380 kV external grid",

    # 150 kV Level
    "b_150_Waterningen": "150 kV Waterningen",
    "b_150_Rijswijk": "150 kV Rijswijk",
    "b_150_sGravenhage": "150 kV 's-Gravenhage",
    "b_150_Voorburg": "150 kV Voorburg",
    "b_150_Krimpen": "150 kV Krimpen",
    "b_150_Ommoord": "150 kV Ommoord",
    "b_150_Zuidwijk": "150 kV Zuidwijk",

    # 25 kV Level
    "b_25_HVS_Zuid": "25 kV HVS Zuid",
    "b_25_Laagveen": "25 kV Laagveen",
    "b_25_HVS_Centrale": "25 kV HVS Centrale",
    "b_25_DH_EB": "25 kV DH 1",
    "b_25_DH": "25 kV DH 2",
    "b_25_DH_CHP": "25 kV DH 3",
    "b_25_HVS_Oost": "25 kV HVS Oost",
    "b_25_Nootdorp2": "25 kV Nootdorp",
    "b_25_Noordsingel": "25 kV Noordsingel",
    "b_25_Ommoord": "25 kV Ommoord",
    "b_25_Bleiswijk": "25 kV Bleiswijk",
    "b_25_RoCa": "25 kV RoCa",

    # 23 kV Level
    "b_23_Zuidwijk": "23 kV Zuidwijk",
    "b_23_RDAM": "23 kV RDAM",
    "b_23_Nootdorp2": "23 kV Nootdorp",

    # 10 kV Level
    "b_10_Laagveen": "10 kV Laagveen",
    "b_10_DH_1": "10 kV DH 1",
    "b_10_HVS_Centrale": "10 kV HVS Centrale",
    "b_10_DH_2": "10 kV DH 2",
    "b_10_Nootdorp2": "10 kV Nootdorp",
    "b_10_Noordsingel": "10 kV Noordsingel",
    "b_10_DH_3": "10 kV DH 3",
    "b_10_DH_4": "10 kV DH 4",
    "b_10_Ommoord": "10 kV Ommoord",
    "b_10_Bleiswijk": "10 kV Bleiswijk",
    "b_10_Tuinders": "10 kV Tuinders",
    "b_10_RDAM": "10 kV RDAM",
}


def plot_power_flow_results(
    bus_result: pd.DataFrame,
    line_result: pd.DataFrame,
    trafo_result: pd.DataFrame,
    experiment_name: str,
    network_path: Path,
    output: Path,
    fig_size: tuple = (15, 8),
    base_bus_result: pd.DataFrame | None = None,
    base_line_result: pd.DataFrame | None = None,
    base_trafo_result: pd.DataFrame | None = None,
) -> None:
    """Plots power flow results, including histograms and a spatial topology map.

    Args:
    -----
        bus_result: DataFrame containing the voltage profile timeseries at each bus.
        line_result: DataFrame containing the line loadings timeseries of each line.
        trafo_result: DataFrame containing the transformer loadings timeseries of each transformer.
        experiment_name: Name of the experiment (used for labeling in comparative histograms).
        network_path: Path to the network's Excel file.
        output: Path to the folder where the output figures will be saved.
        fig_size: Size of the figure for non-spatial plots.
        base_bus_result: DataFrame containing the voltage profile timeseries at each bus for the base case.
        base_line_result: DataFrame containing the line loadings timeseries of each line for the base case.
        base_trafo_result: DataFrame containing the transformer loadings timeseries of each transformer
            for the base case.
    """
    # Clean data to avoid noise and extreme outliers
    # 1. Transformers
    trafo_result_cleaned = trafo_result.copy()
    trafo_result_cleaned[trafo_result_cleaned < 0] = 0
    trafo_result_cleaned.clip(upper=MAX_OVERLOAD, inplace=True)

    # 2. Lines
    line_result_cleaned = line_result.copy()
    line_result_cleaned[line_result_cleaned < 0] = 0
    line_result_cleaned.clip(upper=MAX_OVERLOAD, inplace=True)

    # 3. Buses
    bus_result_cleaned = bus_result.copy()
    bus_result_cleaned[bus_result_cleaned < 0] = 0
    bus_result_cleaned.clip(lower=MIN_VOLTAGE, upper=MAX_VOLTAGE, inplace=True)

    # Visualize spatial results
    if base_bus_result is not None:
        # Plot results with comparison to base case
        logging.info("Plotting relative change in spatial violations compared to base case.")
        # Clean the base case data as well for a fair comparison
        base_bus_result_cleaned = base_bus_result.copy()
        base_bus_result_cleaned[base_bus_result_cleaned < 0] = 0
        base_bus_result_cleaned.clip(lower=MIN_VOLTAGE, upper=MAX_VOLTAGE, inplace=True)

        # Lines
        base_line_result_cleaned = base_line_result.copy()
        base_line_result_cleaned[base_line_result_cleaned < 0] = 0
        base_line_result_cleaned.clip(upper=MAX_OVERLOAD, inplace=True)

        # Transformers
        base_trafo_result_cleaned = base_trafo_result.copy()
        base_trafo_result_cleaned[base_trafo_result_cleaned < 0] = 0
        base_trafo_result_cleaned.clip(upper=MAX_OVERLOAD, inplace=True)

        plot_network_topology_relative_results(
            network_path=network_path,
            bus_result_exp=bus_result_cleaned,
            line_result_exp=line_result_cleaned,
            trafo_result_exp=trafo_result_cleaned,
            bus_result_base=base_bus_result_cleaned,
            line_result_base=base_line_result_cleaned,
            trafo_result_base=base_trafo_result_cleaned,
            output=output,
            bus_names=BUS_NAMES,
        )

        logging.info("Plotting bus, trafo, and line histograms for experiment (base case overlayed for comparison).")
        plot_histograms(
            df1=base_bus_result_cleaned,
            df2=bus_result_cleaned,  # Providing the second DataFrame will plot comparative histograms overlaid
            experiment_name=experiment_name,
            component_type="Bus",
            x_label="Voltage magnitude [p.u.]",
            output=output,
            bus_names=BUS_NAMES,
            components_to_exclude=["380 kV external grid"],
        )
        plot_histograms(
            df1=base_trafo_result_cleaned,
            df2=trafo_result_cleaned,  # Providing the second DataFrame will plot comparative histograms overlaid
            experiment_name=experiment_name,
            component_type="Transformer",
            x_label="Trafo loading [%]",
            output=output,
            trafo_names=TRAFO_NAMES,
        )
        plot_histograms(
            df1=base_line_result_cleaned,
            df2=line_result_cleaned,  # Providing the second DataFrame will plot comparative histograms overlaid
            experiment_name=experiment_name,
            component_type="Line",
            x_label="Line loading [%]",
            output=output,
            line_names=LINE_NAMES,
        )

        logging.info("Plotting transformer and line loading duration curves for experiment.")
        plot_load_duration_curves(
            result_df=trafo_result_cleaned,
            experiment_name=experiment_name,
            base_result_df=base_trafo_result_cleaned,
            component_type="Transformer",
            y_label="Trafo loading [%]",
            output=output,
            trafo_names=TRAFO_NAMES,
            components_to_exclude=None,
        )

        plot_load_duration_curves(
            result_df=line_result_cleaned,
            experiment_name=experiment_name,
            base_result_df=base_line_result_cleaned,
            component_type="Line",
            y_label="Line loading [%]",
            output=output,
            line_names=LINE_NAMES,
            components_to_exclude=None,
        )

    else:
        # Plot results for the base case only (when one dataframe is provided)
        logging.info("Plotting absolute spatial violations for the base case.")
        plot_network_topology_absolute_results(
            network_path=network_path,
            bus_result=bus_result_cleaned,
            line_result=line_result_cleaned,
            trafo_result=trafo_result_cleaned,
            output=output,
            bus_names=BUS_NAMES,
        )
        logging.info("Plotting bus, transformer, and line histograms for the base case.")
        plot_histograms(
            df1=bus_result_cleaned,
            df2=None,  # Providing only only one DataFrame will plot a single histogram
            experiment_name=experiment_name,
            component_type="Bus",
            x_label="Voltage magnitude [p.u.]",
            output=output,
            bus_names=BUS_NAMES,
            components_to_exclude=["380 kV external grid"],
        )
        plot_histograms(
            df1=trafo_result_cleaned,
            df2=None,
            experiment_name=experiment_name,
            component_type="Transformer",
            x_label="Trafo loading [%]",
            output=output,
            trafo_names=TRAFO_NAMES,
        )
        plot_histograms(
            df1=line_result_cleaned,
            df2=None,
            experiment_name=experiment_name,
            component_type="Line",
            x_label="Line loading [%]",
            output=output,
            line_names=LINE_NAMES,
        )

        logging.info("Plotting transformer and line loading duration curves for the base case.")
        plot_load_duration_curves(
            result_df=trafo_result_cleaned,
            experiment_name=experiment_name,
            base_result_df=None,
            component_type="Transformer",
            y_label="Trafo loading [%]",
            output=output,
            trafo_names=TRAFO_NAMES,
            components_to_exclude=None,
        )

        plot_load_duration_curves(
            result_df=line_result_cleaned,
            experiment_name=experiment_name,
            base_result_df=None,
            component_type="Line",
            y_label="Line loading [%]",
            output=output,
            line_names=LINE_NAMES,
            components_to_exclude=None,
        )

    # Visualize temporal results
    logging.info("Plotting temporal results.")
    plot_bus_voltage(bus_result=bus_result_cleaned, fig_size=fig_size, output=output)
    plot_line_loading(line_result=line_result_cleaned, fig_size=fig_size, output=output)
    plot_trafo_loading(trafo_result=trafo_result_cleaned, fig_size=fig_size, output=output)


def plot_network_topology_absolute_results(
    network_path: Path,
    bus_result: pd.DataFrame,
    line_result: pd.DataFrame,
    trafo_result: pd.DataFrame,
    output: Path,
    bus_names: dict | None,
) -> None:
    """Plots the network topology showing absolute violation frequencies (for the base case).

    Args:
    -----
        network_path: Path to the network's Excel file.
        bus_result: DataFrame containing the voltage profile timeseries at each bus.
        line_result: DataFrame containing the voltage profile timeseries at each line.
        trafo_result: DataFrame containing the voltage profile timeseries at each transformer.
        output: Path to the folder where the output figure will be saved.
        bus_names: Dictionary mapping original bus names to new names.
    """
    G = prepare_graph_from_excel(network_path)
    bus_metrics, edge_metrics = calculate_violation_metrics(bus_result, line_result, trafo_result)
    pos = get_custom_network_coordinates()
    bus_freq = bus_metrics.loc[list(G.nodes())]['freq'] * 100  # Convert to percentage
    edge_names = [data['name'] for u, v, data in G.edges(data=True)]
    edge_freq = edge_metrics.loc[edge_names]['freq'] * 100  # Convert to percentage
    edge_types = [data['type'] for u, v, data in G.edges(data=True)]

    node_size = 800
    edge_width = 2.5

    color_palette = ["#11CBBB", "#E80909"]
    cmap_bus = LinearSegmentedColormap.from_list("custom_palette_bus", color_palette, N=256)
    # cmap_bus = cm.get_cmap('plasma')
    norm_bus = Normalize(vmin=0, vmax=100)

    cmap_edge = LinearSegmentedColormap.from_list("custom_palette_edge", color_palette, N=256)
    # cmap_edge = cm.get_cmap('plasma')
    norm_edge = Normalize(vmin=0, vmax=100)

    fig, ax = plt.subplots(figsize=(15, 12))

    lines_edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == 'line']
    trafos_edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == 'transformer']

    line_freqs = [f for f, etype in zip(edge_freq, edge_types) if etype == 'line']
    trafo_freqs = [f for f, etype in zip(edge_freq, edge_types) if etype == 'transformer']

    # Draw network nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes()),
        node_size=node_size,
        node_color=cmap_bus(norm_bus(bus_freq)),
        ax=ax,
    )

    # Draw network edges (full lines for power lines, dashed for transformers)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=lines_edges,
        width=edge_width,
        edge_color=cmap_edge(norm_edge(line_freqs)),
        style='solid',
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=trafos_edges,
        width=edge_width,
        edge_color=cmap_edge(norm_edge(trafo_freqs)),
        style='dashed',
        ax=ax,
    )

    # Draw labels
    if bus_names:
        final_labels = {node: bus_names.get(node, node) for node in G.nodes()}
    else:
        final_labels = {node: node for node in G.nodes()}

    label_offset = 0.15
    pos_labels = {node: (x, y + label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G,
        pos=pos_labels,
        labels=final_labels,
        font_size=10,
        font_weight='bold',
        font_color='black',
    )

    # Add voltage level annotation texts
    level_annotations = {
        '150 kV': 'b_150_Waterningen',
        '25 kV': 'b_25_HVS_Zuid',
        '23 kV': 'b_23_Nootdorp2',
        '10 kV': 'b_10_Laagveen'
    }
    x_annotation = -0.09

    for label_text, bus_name in level_annotations.items():
        if bus_name in pos:
            y_coord = pos[bus_name][1]
            ax.text(
                x_annotation,
                y_coord, label_text,
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                fontweight='bold',
                color='gray'
            )

    # Create custom legend proxies
    bus_proxy = Line2D([], [], color='black', marker='o', linestyle='None', markersize=14, label='Bus')
    line_proxy = Line2D([], [], color='black', linewidth=edge_width, linestyle='solid', label='Power line')
    trafo_proxy = Line2D([], [], color='black', linewidth=edge_width, linestyle='dashed', label='Transformer')
    ax.legend(
        handles=[bus_proxy, line_proxy, trafo_proxy],
        bbox_to_anchor=(0.15, 0.95),
        fontsize='large',
    )

    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the main plot area to make room for color bars

    # Define the position and size of the color bars: [left, bottom, width, height]
    cbar_bus_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])   # Left color bar (superimposed on the right)
    cbar_edge_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Right color bar

    # Create and draw the bus color bar on its dedicated axis
    sm_bus = ScalarMappable(cmap=cmap_bus, norm=norm_bus)
    cbar_bus = plt.colorbar(sm_bus, cax=cbar_bus_ax)
    cbar_bus.ax.yaxis.set_ticks_position('left')
    cbar_bus.ax.yaxis.set_label_position('left')
    cbar_bus.ax.tick_params(labelsize=14)
    cbar_bus.set_label('Bus voltage violation frequency in year  [%]', weight='bold', fontsize=13)

    # Create and draw the edge color bar on its dedicated axis
    sm_edge = ScalarMappable(cmap=cmap_edge, norm=norm_edge)
    cbar_edge = plt.colorbar(sm_edge, cax=cbar_edge_ax)
    cbar_edge.ax.yaxis.set_ticks_position('right')
    cbar_edge.ax.yaxis.set_label_position('right')
    cbar_edge.ax.tick_params(labelsize=14)
    cbar_edge.set_label(
        'Transformer/line loading violation frequency in year  [%]', weight='bold', fontsize=13,   labelpad=13
    )

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(output / "spatial_analysis_absolute.png", dpi=300)
    plt.close(fig)


def plot_network_topology_relative_results(
    network_path: Path,
    bus_result_exp: pd.DataFrame,
    line_result_exp: pd.DataFrame,
    trafo_result_exp: pd.DataFrame,
    bus_result_base: pd.DataFrame,
    line_result_base: pd.DataFrame,
    trafo_result_base: pd.DataFrame,
    output: Path,
    bus_names: dict | None,
) -> None:
    """Plots the network topology showing the relative change in violation frequency compared to base case.

    Args:
    -----
        network_path: Path to the network's Excel file.
        bus_result_exp: Experiment's bus result DataFrame.
        line_result_exp: Experiment's line result DataFrame.
        trafo_result_exp: Experiment's transformer result DataFrame.
        bus_result_base: Base case's bus result DataFrame.
        line_result_base: Base case's line result DataFrame.
        trafo_result_base: Base case's transformer result DataFrame.
        output: Path to the folder where the output figure will be saved.
        bus_names: Dictionary mapping original bus names to new names.
    """
    # Define color map
    # cmap = cm.get_cmap('PiYG', 11)
    color_palette = ["#11CBBB", "#F5F5DC", "#E80909"]
    cmap = LinearSegmentedColormap.from_list("custom_palette", color_palette, N=20)
    max_abs_change = 50  # Maximum absolute change in percentage points for color scaling
    norm = Normalize(vmin=-max_abs_change, vmax=max_abs_change)

    bus_metrics_exp, edge_metrics_exp = calculate_violation_metrics(bus_result_exp, line_result_exp, trafo_result_exp)
    bus_metrics_base, edge_metrics_base = calculate_violation_metrics(
        bus_result_base, line_result_base, trafo_result_base,
    )

    # Calculate relative differences
    bus_freq_diff = (bus_metrics_exp['freq'] - bus_metrics_base['freq']) * 100  # Convert to percentage points
    edge_freq_diff = (edge_metrics_exp['freq'] - edge_metrics_base['freq']) * 100  # Convert to percentage points

    G = prepare_graph_from_excel(network_path)
    pos = get_custom_network_coordinates()

    bus_freq_diff = bus_freq_diff.loc[list(G.nodes())]
    edge_names = [data['name'] for u, v, data in G.edges(data=True)]
    edge_freq_diff = edge_freq_diff.loc[edge_names]
    edge_types = [data['type'] for u, v, data in G.edges(data=True)]

    node_size = 800
    edge_width = 3.0
    casing_width = edge_width + 2.0  # The full width of the black casing

    # Plot
    fig, ax = plt.subplots(figsize=(15, 12))

    lines_edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == 'line']
    trafos_edges = [(u, v) for u, v, data in G.edges(data=True) if data['type'] == 'transformer']

    line_freq_diffs = [f for f, etype in zip(edge_freq_diff, edge_types) if etype == 'line']
    trafo_freq_diffs = [f for f, etype in zip(edge_freq_diff, edge_types) if etype == 'transformer']


    # Draw network nodes with colored casings
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G.nodes()),
        node_size=node_size,
        node_color=cmap(norm(bus_freq_diff)),
        edgecolors='black',
        ax=ax,
    )

    # Draw power lines
    nx.draw_networkx_edges(  # First draw the black casing underneath
        G,
        pos,
        edgelist=lines_edges,
        width=casing_width,
        edge_color='black',
        style='solid',
        ax=ax,
    )
    nx.draw_networkx_edges( # Then draw the colored line on top of the casing
        G,
        pos,
        edgelist=lines_edges,
        width=edge_width,
        edge_color=cmap(norm(line_freq_diffs)),
        style='solid',
        ax=ax,
    )

    # Draw transformers
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=trafos_edges,
        width=casing_width,
        edge_color='black',
        style='dashed',
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=trafos_edges,
        width=edge_width,
        edge_color=cmap(norm(trafo_freq_diffs)),
        style='solid',
        ax=ax,
    )

    # Draw labels
    if bus_names:
        final_labels = {node: bus_names.get(node, node) for node in G.nodes()}
    else:
        final_labels = {node: node for node in G.nodes()}

    label_offset = 0.15
    pos_labels = {node: (x, y + label_offset) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G,
        pos=pos_labels,
        labels=final_labels,
        font_size=10,
        font_weight='bold',
        font_color='black',
        ax=ax,
    )

    # Add voltage level annotation texts
    level_annotations = {
        '150 kV': 'b_150_Waterningen',
        '25 kV': 'b_25_HVS_Zuid',
        '23 kV': 'b_23_Nootdorp2',
        '10 kV': 'b_10_Laagveen'
    }
    x_annotation = -0.09

    for label_text, bus_name in level_annotations.items():
        if bus_name in pos:
            y_coord = pos[bus_name][1]
            ax.text(
                x_annotation,
                y_coord, label_text,
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                fontweight='bold',
                color='gray'
            )

    # Create custom legend
    bus_proxy = Line2D([], [], color='black', marker='o', linestyle='None', markersize=14, label='Bus')
    line_proxy = Line2D([], [], color='black', linewidth=edge_width, linestyle='solid', label='Power line')
    trafo_proxy = Line2D([], [], color='black', linewidth=edge_width, linestyle='dashed', label='Transformer')
    ax.legend(handles=[bus_proxy, line_proxy, trafo_proxy], bbox_to_anchor=(0.15, 0.95), fontsize='large')

    # Adjust layout and add color bars
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_bus_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar_edge_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])

    sm = ScalarMappable(cmap=cmap, norm=norm)

    ticks = [-50, -25, 0, 25, 50]
    cbar_bus = plt.colorbar(sm, cax=cbar_bus_ax)
    cbar_bus.ax.tick_params(labelsize=14)
    cbar_bus.set_ticks(ticks)
    cbar_bus.ax.yaxis.set_ticks_position('left')
    cbar_bus.ax.yaxis.set_label_position('left')
    cbar_bus.set_label(
        'Change in bus voltage violation frequency relative to base case  [%]', weight='bold', fontsize=13,
    )

    cbar_edge = plt.colorbar(sm, cax=cbar_edge_ax)
    cbar_edge.ax.tick_params(labelsize=14)
    cbar_edge.set_ticks(ticks)
    cbar_edge.ax.yaxis.set_ticks_position('right')
    cbar_edge.ax.yaxis.set_label_position('right')
    cbar_edge.set_label(
        'Change in transformer/line loading violation frequency relative to base case [%]', weight='bold', fontsize=13,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(output / "spatial_analysis_relative.png", dpi=300)
    plt.close(fig)


def plot_bus_voltage(bus_result: pd.DataFrame, fig_size: tuple, output: Path) -> None:
    """Plots bus voltages for different branches in the electricity network.

    Args:
    -----
        bus_result: DataFrame containing the voltage profile timeseries at each bus.
        fig_size: Size of the figure to create.
        output: Path to the folder where the output figures will be saved.
    """
    HV_BUS = {  # 150 kV Level
        "b_150_Waterningen": {"new_name": BUS_NAMES["b_150_Waterningen"], "color": "#BE06BE"},
        "b_150_Rijswijk": {"new_name": BUS_NAMES["b_150_Rijswijk"], "color": "#ff5602"},
        "b_150_sGravenhage": {"new_name": BUS_NAMES["b_150_sGravenhage"], "color": "#01c8ff"},
        "b_150_Voorburg": {"new_name": BUS_NAMES["b_150_Voorburg"], "color": "#0a8457"},
        "b_150_Krimpen": {"new_name": BUS_NAMES["b_150_Krimpen"], "color": "#0101ff"},
        "b_150_Ommoord": {"new_name": BUS_NAMES["b_150_Ommoord"], "color": "#ffdd00"},
        "b_150_Zuidwijk": {"new_name": BUS_NAMES["b_150_Zuidwijk"], "color": "#ff00aa"},
    }

    MV_BUS = {  # 25 kV and 23 kV Level
        "b_25_HVS_Zuid": {"new_name": BUS_NAMES["b_25_HVS_Zuid"], "color": "#E32DE3"},
        "b_25_Laagveen": {"new_name": BUS_NAMES["b_25_Laagveen"], "color": "#ff5602"},
        "b_25_HVS_Centrale": {"new_name": BUS_NAMES["b_25_HVS_Centrale"], "color": "#01c8ff"},
        "b_25_DH_EB": {"new_name": BUS_NAMES["b_25_DH_EB"], "color": "#0a8457"},
        "b_25_DH": {"new_name": BUS_NAMES["b_25_DH"], "color": "#0101ff"},
        "b_25_DH_CHP": {"new_name": BUS_NAMES["b_25_DH_CHP"], "color": "#ffdd00"},
        "b_25_HVS_Oost": {"new_name": BUS_NAMES["b_25_HVS_Oost"], "color": "#ff00aa"},
        "b_25_Nootdorp2": {"new_name": BUS_NAMES["b_25_Nootdorp2"], "color": "#00ffaa"},
        "b_25_Noordsingel": {"new_name": BUS_NAMES["b_25_Noordsingel"], "color": "#5b0984"},
        "b_25_Ommoord": {"new_name": BUS_NAMES["b_25_Ommoord"], "color": "#ffaa00"},
        "b_25_Bleiswijk": {"new_name": BUS_NAMES["b_25_Bleiswijk"], "color": "#00aaff"},
        "b_25_RoCa": {"new_name": BUS_NAMES["b_25_RoCa"], "color": "#aaff00"},
        "b_23_Zuidwijk": {"new_name": BUS_NAMES["b_23_Zuidwijk"], "color": "#ff0066"},
        "b_23_RDAM": {"new_name": BUS_NAMES["b_23_RDAM"], "color": "#66ff00"},
        "b_23_Nootdorp2": {"new_name": BUS_NAMES["b_23_Nootdorp2"], "color": "#0066ff"},
    }

    LV_BUS = {  # 10 kV Level
        "b_10_Laagveen": {"new_name": BUS_NAMES["b_10_Laagveen"], "color": "#E32DE3"},
        "b_10_DH_1": {"new_name": BUS_NAMES["b_10_DH_1"], "color": "#ff5602"},
        "b_10_HVS_Centrale": {"new_name": BUS_NAMES["b_10_HVS_Centrale"], "color": "#01c8ff"},
        "b_10_DH_2": {"new_name": BUS_NAMES["b_10_DH_2"], "color": "#0a8457"},
        "b_10_Nootdorp2": {"new_name": BUS_NAMES["b_10_Nootdorp2"], "color": "#0101ff"},
        "b_10_Noordsingel": {"new_name": BUS_NAMES["b_10_Noordsingel"], "color": "#ff0011"},
        "b_10_DH_3": {"new_name": BUS_NAMES["b_10_DH_3"], "color": "#ff00aa"},
        "b_10_DH_4": {"new_name": BUS_NAMES["b_10_DH_4"], "color": "#00ffaa"},
        "b_10_Ommoord": {"new_name": BUS_NAMES["b_10_Ommoord"], "color": "#5b0984"},
        "b_10_Bleiswijk": {"new_name": BUS_NAMES["b_10_Bleiswijk"], "color": "#ffaa00"},
        "b_10_Tuinders": {"new_name": BUS_NAMES["b_10_Tuinders"], "color": "#00aaff"},
        "b_10_RDAM": {"new_name": BUS_NAMES["b_10_RDAM"], "color": "#aaff00"},
    }

    configs = [
        {"buses": HV_BUS, "filename": "bus_voltage_HV.png", "use_custom_label": True},
        {"buses": MV_BUS, "filename": "bus_voltage_MV.png", "use_custom_label": True},
        {"buses": LV_BUS, "filename": "bus_voltage_LV.png", "use_custom_label": True},
    ]

    for cfg in configs:
        # Filter for buses that are actually in our results
        buses_to_plot = [bus for bus in cfg["buses"] if bus in bus_result.columns]
        if not buses_to_plot:
            print(f"Skipping {cfg['filename']}, no matching buses found.")
            continue

        # Prepare ordered lists of colors and new names for the legend
        colors_for_plot = [cfg["buses"][bus]["color"] for bus in buses_to_plot]

        if cfg["use_custom_label"]:
            labels_for_plot = [cfg["buses"][bus]["new_name"] for bus in buses_to_plot]
        else:
            labels_for_plot = buses_to_plot

        ax = bus_result[buses_to_plot].plot(
            figsize=fig_size,
            grid=False,
            color=colors_for_plot,
            legend=False
        )

        customize_and_save_plot(
            ax=ax,
            output_dir=output,
            filename=cfg["filename"],
            ylabel="Bus voltage [p.u]",
            title="",
            add_grid=True,
            y_axis_formatter="%.2f",
            add_legend=True,
            legend_title="Bus",
            legend_handles=ax.get_lines(),
            legend_labels=labels_for_plot,
            tight_layout_rect=[0, 0, 0.85, 1],
            bbox_to_anchor=(1.0, 0.95),
        )


def plot_line_loading(line_result: pd.DataFrame, fig_size: tuple, output: Path) -> None:
    """Plots line loadings for different branches in the electricity network.

    Args:
    -----
        line_result: DataFrame containing the line loadings timeseries of each line.
        fig_size: Size of the figure to create.
        output: Path to the folder where the output figures will be saved.
    """
    HV_LINES = {
        "150kV_Waterningen_Rijskwijk": {"new_name": LINE_NAMES["150kV_Waterningen_Rijskwijk"], "color": "#99d921"},
        "150kV_sGravenhage_Rijskwijk": {"new_name": LINE_NAMES["150kV_sGravenhage_Rijskwijk"], "color": "#268409"},
        "150kV_sGravenhage_Voorburg": {"new_name": LINE_NAMES["150kV_sGravenhage_Voorburg"], "color": "#ff0202"},
        "150kV_Voorburg_Waterningen": {"new_name": LINE_NAMES["150kV_Voorburg_Waterningen"], "color": "#ff9e02"},
        "150kV_Krimpen_Ommoord(1)": {"new_name": LINE_NAMES["150kV_Krimpen_Ommoord(1)"], "color": "#02ffd9"},
        "150kV_Krimpen_Ommoord(2)": {"new_name": LINE_NAMES["150kV_Krimpen_Ommoord(2)"], "color": "#ff02f7"},
        "150kV_Krimpen_Zuidwijk": {"new_name": LINE_NAMES["150kV_Krimpen_Zuidwijk"], "color": "#020aff"},
    }

    MV_LINES = {
        "25kV_HVS_Zuid_Laagveen(1)": {"new_name": LINE_NAMES["25kV_HVS_Zuid_Laagveen(1)"], "color": "#99d921"},
        "25kV_HVS_Zuid_Laagveen(2&3)": {"new_name": LINE_NAMES["25kV_HVS_Zuid_Laagveen(2&3)"], "color": "#268409"},
        "25kV_HVS_Zuid_Laagveen(4)": {"new_name": LINE_NAMES["25kV_HVS_Zuid_Laagveen(4)"], "color": "#dd1616"},
        "25kV_HVS_Centrale_DH_EB": {"new_name": LINE_NAMES["25kV_HVS_Centrale_DH_EB"], "color": "#ff9e02"},
        "25kV_HVS_Centrale_DH_HP": {"new_name": LINE_NAMES["25kV_HVS_Centrale_DH_HP"], "color": "#02ffd9"},
        "25kV_DH_CHP_Cables": {"new_name": LINE_NAMES["25kV_DH_CHP_Cables"], "color": "#f52fee"},
        "25kV_HVS_Oost_Nootdorp2": {"new_name": LINE_NAMES["25kV_HVS_Oost_Nootdorp2"], "color": "#020aff"},
        "25kV_HVS_Oost_Noordsingel(1)": {"new_name": LINE_NAMES["25kV_HVS_Oost_Noordsingel(1)"], "color": "#0F7C6D"},
        "25kV_HVS_Oost_Noordsingel(2)": {"new_name": LINE_NAMES["25kV_HVS_Oost_Noordsingel(2)"], "color": "#9a6209"},
        "25kV_Ommoord_RoCa(1&2)": {"new_name": LINE_NAMES["25kV_Ommoord_RoCa(1&2)"], "color": "#3d74d4"},
        "25kV_Ommoord_Bleiswijk(1&2)": {"new_name": LINE_NAMES["25kV_Ommoord_Bleiswijk(1&2)"], "color": "#760f9b"},
        "23kV_Zuidwijk_RDAM": {"new_name": LINE_NAMES["23kV_Zuidwijk_RDAM"], "color": "#21d96e"},
    }

    LV_LINES = {
        "10kV_Laagveen_DH_1": {"new_name": LINE_NAMES["10kV_Laagveen_DH_1"], "color": "#99d921"},
        "10kV_HVS_Centrale_DH_2": {"new_name": LINE_NAMES["10kV_HVS_Centrale_DH_2"], "color": "#ff5602"},
        "10kV_Nootdorp2_DH_3": {"new_name": LINE_NAMES["10kV_Nootdorp2_DH_3"], "color": "#01c8ff"},
        "10kV_Noordsingel_DH_4": {"new_name": LINE_NAMES["10kV_Noordsingel_DH_4"], "color": "#02ffd9"},
        "10kV_Ommoord_RDAM": {"new_name": LINE_NAMES["10kV_Ommoord_RDAM"], "color": "#f52fee"},
        "10kV_Bleiswijk_Tuinders": {"new_name": LINE_NAMES["10kV_Bleiswijk_Tuinders"], "color": "#020aff"},
    }

    configs = [
        {"lines": HV_LINES, "filename": "line_loading_HV.png", "use_custom_label": True},
        {"lines": MV_LINES, "filename": "line_loading_MV.png", "use_custom_label": True},
        {"lines": LV_LINES, "filename": "line_loading_LV.png", "use_custom_label": True},
    ]

    for cfg in configs:
        # Filter for lines that are actually in our results
        lines_to_plot = [line for line in cfg["lines"] if line in line_result.columns]
        if not lines_to_plot:
            print(f"Skipping {cfg['filename']}, no matching lines found.")
            continue

        # Prepare ordered lists of colors and new names for the legend
        colors_for_plot = [cfg["lines"][line]["color"] for line in lines_to_plot]

        if cfg["use_custom_label"]:
            labels_for_plot = [cfg["lines"][line]["new_name"] for line in lines_to_plot]
        else:
            labels_for_plot = lines_to_plot

        ax = line_result[lines_to_plot].plot(
            figsize=fig_size,
            grid=False,
            color=colors_for_plot,
            legend=False
        )

        customize_and_save_plot(
            ax=ax,
            output_dir=output,
            filename=cfg["filename"],
            ylabel="Line loading [%]",
            title="",
            add_grid=True,
            y_axis_formatter="%.1f",
            add_legend=True,
            legend_title="Power Lines",
            legend_handles=ax.get_lines(),
            legend_labels=labels_for_plot,
            tight_layout_rect=[0, 0, 0.8, 1],
            bbox_to_anchor=(1.0, 0.95),
        )


def plot_trafo_loading(trafo_result: pd.DataFrame, fig_size: tuple, output: Path) -> None:
    """Plots transformer loadings for different branches in the electricity network.

    Args:
    -----
        trafo_result: DataFrame containing the transformer loadings timeseries of each transformer.
        fig_size: Size of the figure to create.
        output: Path to the folder where the output figures will be saved.
    """
    TRAFO_HV = {  # 380/150kV transformers
        "T_Waterningen": {"new_name": TRAFO_NAMES["T_Waterningen"], "color":  "#BE06BE"},
        "T_Krimpen": {"new_name": TRAFO_NAMES["T_Krimpen"], "color": "#01c8ff"},
    }
    TRAFO_HV_MV = {  # 150/25kV and 150/23kV transformers
        "T_Rijswijk": {"new_name": TRAFO_NAMES["T_Rijswijk"], "color": "#BE06BE"},
        "T_sGravenhage": {"new_name": TRAFO_NAMES["T_sGravenhage"], "color": "#ff5602"},
        "T_Voorburg": {"new_name": TRAFO_NAMES["T_Voorburg"], "color": "#01c8ff"},
        "T_Ommoord_HV": {"new_name": TRAFO_NAMES["T_Ommoord_HV"], "color": "#0a8457"},
        "T_Zuidwijk": {"new_name": TRAFO_NAMES["T_Zuidwijk"], "color": "#0101ff"},
    }
    TRAFO_MV_LV = {  # 25/10kV and 25/23kV transformers
        "T_Laagveen": {"new_name": TRAFO_NAMES["T_Laagveen"], "color": "#99d921"},
        "T_HVS_Centrale": {"new_name": TRAFO_NAMES["T_HVS_Centrale"], "color": "#ff0202"},
        "T_Nootdorp2_MV1": {"new_name": TRAFO_NAMES["T_Nootdorp2_MV1"], "color": "#01c8ff"},
        "T_Nootdorp2_MV2": {"new_name": TRAFO_NAMES["T_Nootdorp2_MV2"], "color": "#ff7f0e"},
        "T_Noordsingel": {"new_name": TRAFO_NAMES["T_Noordsingel"], "color": "#0a8457"},
        "T_Ommoord_MV": {"new_name": TRAFO_NAMES["T_Ommoord_MV"], "color": "#BE06BE"},
        "T_Bleiswijk1": {"new_name": TRAFO_NAMES["T_Bleiswijk1"], "color": "#0101ff"},
    }

    configs = [
        {
            "group": TRAFO_HV,
            "filename": "transformer_loading_HV.png",
            "use_custom_label": True,  # If True, use `new_name` for legend labels
            "sort_legend": False,
        },
        {
            "group": TRAFO_HV_MV,
            "filename": "transformer_loading_HV_MV.png",
            "use_custom_label": True,
            "sort_legend": False,
        },
        {
            "group": TRAFO_MV_LV,
            "filename": "transformer_loading_MV_LV.png",
            "use_custom_label": True,
            "sort_legend": True,
        },
    ]

    for cfg in configs:
        trafos_to_plot = [trafo for trafo in cfg["group"] if trafo in trafo_result.columns]
        if not trafos_to_plot:
            print(f"Skipping {cfg['filename']}, no matching transformers found.")
            continue

        # Prepare colors and custom labels for the legend
        colors = []
        custom_labels_map = {} # Maps original column name to desired/encoded names for legend display
        for trafo in trafos_to_plot:
            trafo_meta = cfg["group"].get(trafo, {})
            colors.append(trafo_meta.get("color"))

            # Determine the label for the legend
            if cfg["use_custom_label"]:
                display_name = trafo_meta.get("new_name", trafo)
                custom_labels_map[trafo] = f"{display_name}"
            else:
                custom_labels_map[trafo] = trafo

        # Plot the data, passing the custom colors.
        ax = trafo_result[trafos_to_plot].plot(figsize=fig_size, grid=False, legend=False, color=colors)

        # Customize the legend labels
        # Get the plot handles and the original labels (which are the column names) from the axes
        handles, original_labels = ax.get_legend_handles_labels()

        # Create the final list of labels using the custom_labels_map
        final_labels = [custom_labels_map[label] for label in original_labels]

        # Sort the handles and labels together if required
        if cfg["sort_legend"]:
            sorted_pairs = sorted(zip(final_labels, handles), key=lambda pair: pair[0])
            final_labels, handles = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Call the customize_and_save_plot function with the final handles and labels
        customize_and_save_plot(
            ax=ax,
            output_dir=output,
            filename=cfg["filename"],
            ylabel="Transformer loading [%]",
            title="",
            add_grid=True,
            y_axis_formatter="%.1f",
            add_legend=True,
            legend_title="Transformer",
            legend_handles=list(handles),
            legend_labels=list(final_labels),
            tight_layout_rect=[0, 0, 0.84, 1],
            bbox_to_anchor=(1.0, 0.95),
        )


def plot_histograms(
    df1: pd.DataFrame,
    component_type: str,
    x_label: str,
    output: Path,
    experiment_name: str,
    df2: pd.DataFrame | None = None,
    font_size: int = 10,
    bus_names: dict | None = None,
    line_names: dict | None = None,
    trafo_names: dict | None = None,
    components_to_exclude: list[str] | None = None,
) -> None:
    """Plots histograms of power flow results for buses, lines, or transformers.

    Args:
    -----
        df1: DataFrame containing the primary power flow results (base case).
        component_type: Type of component ("Bus", "Line", or "Transformer").
        x_label: Label for the x-axis of the histogram.
        output: Path to the folder where the output figures will be saved.
        experiment_name: Name of the experiment (used in legend if df2 is provided).
        df2: Optional DataFrame with the secondary results (will be used to compare against df1).
        font_size: Font size for the plot text elements.
        bus_names: Dictionary mapping original bus names to new names.
        line_names: Dictionary mapping original line names to new names.
        trafo_names: Dictionary mapping original transformer names to new names.
        components_to_exclude: List of component new names (from the renamed columns) to exclude from the plots.
    """
    # Define a safe limit for the number of bins to avoid memory issues
    MAX_BINS = 50

    # Define colors for the plots
    base_case_color = "#10DECD"
    exp_case_color = "#E84F4F"
    voltage_limit_color = "#E80909"

    # Create renaming map based on component type
    rename_map_lookup = {"Bus": bus_names, "Line": line_names, "Transformer": trafo_names}
    rename_map = rename_map_lookup[component_type]

    # Process first DataFrame (df1)
    base_df = df1.rename(columns=rename_map)
    base_df = sort_result_df_columns_by_voltage_levels(base_df)
    if components_to_exclude:
        base_df = base_df.drop(columns=components_to_exclude, errors='ignore')
        # Raise a warning if any specified components are not found
        missing_components = set(components_to_exclude) - set(df1.rename(columns=rename_map).columns)
        if missing_components:
            logging.warning(f"⚠️The {missing_components=} were not found in the result DataFrame.")

    # Process second DataFrame (df2), if it exists
    exp_df = None
    if df2 is not None:
        exp_df = df2.rename(columns=rename_map)
        exp_df = sort_result_df_columns_by_voltage_levels(exp_df)
        if components_to_exclude:
            exp_df = exp_df.drop(columns=components_to_exclude, errors='ignore')

    components = base_df.columns
    num_components = len(components)
    if num_components == 0:
        logging.warning(f"No components of type {component_type} found to plot.")
        return

    # Calculate the grid size for the subplots to be as square as possible
    ncols = math.ceil(math.sqrt(num_components))
    nrows = math.ceil(num_components / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3.5))
    ax_flat = axes.flatten()

    # Loop through each component to create a subplot
    for i, component_name in enumerate(components):
        ax = ax_flat[i]

        # Determine combined bin range to align histograms
        min_val, max_val = base_df[component_name].min(), base_df[component_name].max()
        if exp_df is not None and component_name in exp_df.columns:
            min_val = min(min_val, exp_df[component_name].min())
            max_val = max(max_val, exp_df[component_name].max())
        num_bins = np.linspace(min_val, max_val, MAX_BINS + 1)

        # Plot first histogram (always)
        sns.histplot(
            x=base_df[component_name],
            bins=num_bins,
            kde=False,
            ax=ax,
            color=base_case_color,
            stat="percent",
            alpha=0.7,
        )

        # Plot second histogram if it exists
        if exp_df is not None and component_name in exp_df.columns:
            sns.histplot(
                x=exp_df[component_name],
                bins=num_bins,
                kde=False,
                ax=ax,
                color=exp_case_color,
                stat="percent",
                alpha=0.7,
            )

        # Customize KDE lines and legend
        if exp_df is not None and component_name in exp_df.columns:
            handles = [
                Patch(color=base_case_color, alpha=0.7, label='Base case'),
                Patch(color=exp_case_color, alpha=0.7, label=experiment_name)
            ]
            ax.legend(handles=handles, fontsize=font_size)

        # Set titles and labels
        ax.set_title(component_name, fontsize=font_size + 1)
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel("Fraction of hours in year  [%]", fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)

        # Set x-axis formatter based on component type and data spread
        if component_type == "Bus":
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

            # Also add 2 vertical lines at 0.95 p.u. and 1.05 p.u. to show voltage limits
            ax.axvline(x=0.95, color=voltage_limit_color, linestyle='--', linewidth=1.5)
            ax.axvline(x=1.05, color=voltage_limit_color, linestyle='--', linewidth=1.5)

        elif component_type in ["Line", "Transformer"]:
            data_range = (max_val - min_val)
            if data_range > 1.0:
                formatter = FormatStrFormatter('%.0f')
            else:
                formatter = FormatStrFormatter('%.2f')
            ax.xaxis.set_major_formatter(formatter)

            # Add vertical line at 100% to show loading limit
            ax.axvline(x=100.0, color=voltage_limit_color, linestyle='--', linewidth=1.5)

    # Hide any unused subplots
    for i in range(num_components, len(ax_flat)):
        ax_flat[i].set_visible(False)

    fig.tight_layout()
    plt.savefig(output / f"{component_type.lower()}_histogram.png", dpi=300)
    plt.close(fig)


def plot_load_duration_curves(
    result_df: pd.DataFrame,
    experiment_name: str,
    component_type: str,
    y_label: str,
    output: Path,
    font_size: int = 10,
    base_result_df: pd.DataFrame | None = None,
    bus_names: dict | None = None,
    line_names: dict | None = None,
    trafo_names: dict | None = None,
    components_to_exclude: list[str] | None = None,
) -> None:
    """Plots load duration curves of power flow results for buses, lines, or transformers.

    Args:
    -----
        result_df: DataFrame containing the power flow results timeseries.
        experiment_name: Experiment name used in legend for comparison if second df is provided.
        component_type: Type of component ("Bus", "Line", or "Transformer").
        y_label: Label for the y-axis of the plot.
        output: Path to the folder where the output figures will be saved.
        font_size: Font size for titles and labels.
        base_result_df: DataFrame with the base case results for overlay.
        bus_names: Dictionary mapping original bus names to new names.
        line_names: Dictionary mapping original line names to new names.
        trafo_names: Dictionary mapping original transformer names to new names.
        components_to_exclude: List of component new names to exclude.
    """
    # exp_case_color = "#e253b0"
    exp_case_color = "#E17A7A"
    base_case_line_color = "#0EA68A"
    loading_limit_color = "#E80909"

    # Rename columns and sort by voltage levels
    rename_map_lookup = {"Bus": bus_names, "Line": line_names, "Transformer": trafo_names}
    rename_map = rename_map_lookup[component_type]
    df_renamed = result_df.rename(columns=rename_map)
    df_to_plot = sort_result_df_columns_by_voltage_levels(df_renamed)

    base_df_renamed = None
    if base_result_df is not None:
        base_df_renamed = base_result_df.rename(columns=rename_map)

    # Exclude specified components if any
    if components_to_exclude:
        df_to_plot = df_to_plot.drop(columns=components_to_exclude, errors='ignore')
        # Raise a warning if any specified components are not found in the result DataFrame
        missing = set(components_to_exclude) - set(df_renamed.columns)
        if missing:
            logging.warning(f"⚠️ Components to exclude not found in result_df: {missing}")
        if base_df_renamed is not None:
            base_df_renamed = base_df_renamed.drop(columns=components_to_exclude, errors='ignore')
            missing_base = set(components_to_exclude) - set(base_df_renamed.columns)
            if missing_base:
                logging.warning(f"⚠️ Components to exclude not found in base_result_df: {missing_base}")

    components = df_to_plot.columns
    num_components = len(components)

    # Calculate the grid size for the subplots to be as square as possible
    ncols = math.ceil(math.sqrt(num_components))
    nrows = math.ceil(num_components / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3.5))
    ax_flat = axes.flatten()

    # Loop through each component to create a subplot
    for i, component_name in enumerate(components):
        ax = ax_flat[i]

        # Sort data in descending order for the duration curve
        sorted_data = df_to_plot[component_name].sort_values(ascending=False).values
        x_values = np.arange(len(sorted_data))

        # Plot the experiment case as a filled area chart
        ax.fill_between(x_values, sorted_data, color=exp_case_color, alpha=0.8, label=experiment_name)

        # Overlay the base case load duration curve line if provided
        if base_df_renamed is not None and component_name in base_df_renamed.columns:
            base_sorted_data = base_df_renamed[component_name].sort_values(ascending=False).values
            ax.plot(x_values, base_sorted_data, color=base_case_line_color, linewidth=3, label="Base case")
            ax.legend(fontsize=font_size) # Show legend only when comparing
            # Reverse legend order to have base case on top
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], fontsize=font_size)

        # Set titles, labels, and axis limits
        ax.set_title(component_name, fontsize=font_size + 1)
        ax.set_xlabel("Duration  [hours]", fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xlim(0, len(x_values))

        if component_type == "Bus":
            ax.set_ylim(MIN_VOLTAGE, MAX_VOLTAGE)
        elif component_type in ["Line", "Transformer"]:
            ax.set_ylim(0, MAX_OVERLOAD)

            # Add horizontal line at 100% to show loading limit
            ax.axhline(y=100.0, color=loading_limit_color, linestyle='--', linewidth=1.5)

    # Hide any unused subplots
    for j in range(num_components, len(ax_flat)):
        ax_flat[j].set_visible(False)

    fig.tight_layout()
    plt.savefig(output / f"{component_type.lower()}_duration_curve.png", dpi=300)
    plt.close(fig)
