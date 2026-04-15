import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import pypsa

PYPSA_COMPONENTS = {"Generator", "Link", "StorageUnit", "Store"}

# Set of carriers of electricity producing links: produces heat at bus1 (p1) and electricity at bus2 (p2).
CARRIERS_ELEC_PROD_LINKS = {"CHP-greengas", "CHP-hydrogen"}

# Set of carriers of electricity consuming links
CARRIERS_ELEC_CONS_LINKS = {"ASHP", "Electric boiler", "Geothermal ASHP", "HT-ATES-ASHP"}

UNDER_VOLTAGE_START = 0.94
OVER_VOLTAGE_START = 1.06
OVERLOAD_START = 110.0

UNDER_VOLTAGE_SEVERE = UNDER_VOLTAGE_START - 0.001  # 0.939
OVER_VOLTAGE_SEVERE = OVER_VOLTAGE_START + 0.001    # 1.061
OVERLOAD_SEVERE = OVERLOAD_START + 0.5              # 110.5

SUSTAINED_VIOLATION_STREAK_HOURS = 7  # number of hours to consider a violation  "sustained/problematic"


def get_assets_based_on_carrier_name(n: pypsa.Network, component_type: str, carrier_name: str) -> list[str]:
    """Get a list of assets in the network based on their carrier name.

    Args:
    -----
        n: PyPSA network object.
        component_type: Type of the component to search in (e.g., "Generator", "Link", "Store").
        carrier_name: Carrier name to search for in the asset names.

    Returns:
    -------
        List of assets matching the carrier name.

    Raises:
    -------
        ValueError: If the component type is not recognized.
    """
    if component_type not in PYPSA_COMPONENTS:
        raise ValueError(f"Unsupported component type: {component_type}. Supported types: {PYPSA_COMPONENTS}")

    if component_type == "Generator":
        assets = [gn for gn in n.generators.index if n.generators.loc[gn, "carrier"].lower() == carrier_name.lower()]
    elif component_type == "Link":
        assets = [lk for lk in n.links.index if n.links.loc[lk, "carrier"].lower() == carrier_name.lower()]
    elif component_type == "StorageUnit":
        assets = [
            su for su in n.storage_units.index if n.storage_units.loc[su, "carrier"].lower() == carrier_name.lower()
        ]
    elif component_type == "Store":
        assets = [st for st in n.stores.index if n.stores.loc[st, "carrier"].lower() == carrier_name.lower()]

    if not assets:
        warnings.warn(f"No assets found in the network matching carrier name: {carrier_name}")

    return assets


def get_electricity_generation_of_assets(n: pypsa.Network) -> pd.DataFrame:
    """Get the electricity generation of all assets in the network.

    This includes CHPs that generate electricity as an output.

    Args:
    -----
        n: PyPSA network object.

    Returns:
    -------
        DataFrame with electricity generation of all relevant assets.
    """
    carriers_all_links = set(n.links.carrier)
    elec_producing_assets = []

    for carrier in CARRIERS_ELEC_PROD_LINKS:
        if carrier not in carriers_all_links:
            raise ValueError(f"Carrier {carrier} not found in the network links.")
        elec_producing_assets.extend(
            get_assets_based_on_carrier_name(n, component_type="Link", carrier_name=carrier)
        )

    electricity_production_df = n.links_t.p2.reindex(columns=elec_producing_assets, fill_value=0)

    # Multiply by -1 since in PyPSA, link power is negative at bus where it is generating (instead of consuming) power.
    electricity_production_df *= -1

    return electricity_production_df


def get_electricity_consumption_of_assets(n: pypsa.Network) -> pd.DataFrame:
    """Get the electricity consumption of all assets in the network.

    This includes heat pumps, electric boilers, geothermal boosters,
    and HT-ATES heat pump units that have electricity as an input.

    Args:
    -----
        n: PyPSA network object.

    Returns:
    -------
        DataFrame with electricity consumption of all relevant assets.
    """
    carriers_all_links = set(n.links.carrier)
    elec_consuming_assets = []

    for carrier in (CARRIERS_ELEC_CONS_LINKS):
        if carrier not in carriers_all_links:
            raise ValueError(f"Carrier {carrier} not found in the network links.")
        elec_consuming_assets.extend(
            get_assets_based_on_carrier_name(n, component_type="Link", carrier_name=carrier)
        )

    electricity_consumption_df = n.links_t.p0.reindex(columns=elec_consuming_assets, fill_value=0)

    return electricity_consumption_df


def fill_path_wildcards(path_template: str | Path, params: dict) -> Path:
    """Fill in wildcards in the path template with values from the params dict.

    Args:
    -----
        path_template: The path template with wildcards to be filled.
        params: A dictionary of parameters to fill in the wildcards.

    Returns:
    --------
        Filled path with wildcards replaced by parameter values.

    Example:
    --------
        filled_path = "data/{year}/{month}/file.csv"
        params = {"year": 2023, "month": 3}
        path = fill_path_wildcards(filled_path, params)
        # Result: Path("data/2023/3/file.csv")
    """
    if not isinstance(path_template, str):
        path_template = str(path_template)

    try:
        return Path(path_template.format(**params))
    except KeyError as e:
        raise ValueError(f"Missing parameter in data source path: {e}. Provided parameters were: {params}")


def calculate_heatpump_cop(tech_carrier: str, temp_source: np.ndarray) -> np.ndarray:
    """Calculate the COP of heat pumps based on ambient and sink temperatures.

    This formula follows from PyPSA documentation:
    https://pypsa.readthedocs.io/en/latest/examples/sector-coupling-single-node.html#Heat-pumps
    https://pypsa-eur.readthedocs.io/en/latest/supply_demand.html#heat-supply

    Args:
    -----
        tech_carrier: The carrier/type of the technology ("ASHP", "GSHP").
        temp_source: The temperature of the heat source.
        temp_sink: The temperature of the heat sink.

    Returns:
    --------
        The COP of the technology as a numpy array.

    Raises:
        ValueError: If the technology type is not recognized.
    """
    tech_carrier = tech_carrier.lower()
    TECHS_WITH_TIME_VARYING_COP = {
    "ashp": {"a": 6.81, "b": -0.121, "c": 0.000630, "temp_sink": 65},
    "gshp": {"a": 8.77, "b": -0.150, "c": 0.000734, "temp_sink": 45},
}
    if tech_carrier not in TECHS_WITH_TIME_VARYING_COP:
        raise ValueError(f"Unrecognized {tech_carrier=}. Supported: {TECHS_WITH_TIME_VARYING_COP.keys()=}")

    tech_attr = TECHS_WITH_TIME_VARYING_COP[tech_carrier]
    temp_diff = tech_attr["temp_sink"] - temp_source
    coef_of_performance = tech_attr["a"] + tech_attr["b"] * temp_diff + tech_attr["c"] * temp_diff**2

    return coef_of_performance


def customize_and_save_plot(
    ax: matplotlib.axes.Axes,
    output_dir: Path,
    filename: str,
    ylabel: str,
    xlabel: str = "",
    title: str = "",
    fontsize: float = 15,
    add_grid: bool = True,
    y_axis_formatter: str = "%.0f",
    ylim: tuple[float, float] = None,
    add_legend: bool = True,
    legend_title: str = None,
    legend_labels: list = None,
    legend_handles: list = None,
    tight_layout_rect: list | None = None,
    bbox_to_anchor: tuple = None,
) -> None:
    """Customize figure labels, titles, etc., and save the figure.

    Args:
    -----
        ax: The axes object to customize and save.
        output_dir: Path to the folder where the output figure will be saved.
        filename: Name of the output file (must end with an extension: .png, .pdf, etc.)
        ylabel: Label for the y-axis.
        xlabel: Label for the x-axis.
        title: Title of the plot.
        fontsize: Font size for the plot.
        add_grid: Whether to add a grid to the plot.
        y_axis_formatter: Formatter string for the y-axis ticks.
        ylim: Tuple specifying y-axis limits (min, max). If None, defaults are used.
        add_legend: Whether or not to add a legend to the plot.
        legend_title: Title for the legend.
        legend_handles: List of custom legend handles.
        legend_labels: List of custom legend labels.
        tight_layout_rect: Optional rectangle (left, bottom, right, top) to pass to tight_layout.
        bbox_to_anchor: Optional tuple to specify legend bbox_to_anchor parameter.
    """
    ax.set_xlabel(xlabel, fontsize=fontsize-3)
    ax.set_ylabel(ylabel, fontsize=fontsize-2, labelpad=10)
    ax.tick_params(axis="y", labelsize=fontsize-2)
    ax.tick_params(axis="x", labelsize=fontsize-3)

    if title:
        ax.set_title(title, fontsize=fontsize + 1)

    if add_grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Format y-axis ticks using the provided formatter string
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(y_axis_formatter))

    # Get the figure object from the axes to save it
    fig = ax.get_figure()
    if tight_layout_rect:
        fig.tight_layout(rect=tight_layout_rect)
    else:
        fig.tight_layout()

    # Legend customizations
    legend_fontsize = fontsize - 4
    legend_title_fontsize = fontsize - 2.5
    if add_legend:
        # If custom handles and labels are provided, use them.
        if legend_handles and legend_labels:
            legend = ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                title=legend_title,
                fontsize=legend_fontsize,
                bbox_to_anchor=bbox_to_anchor,
            )
        # Otherwise, let matplotlib create the default legend.
        else:
            legend = ax.legend(title=legend_title, fontsize=legend_fontsize, bbox_to_anchor=bbox_to_anchor)

        # Adjust legend title font size if title is provided
        if legend and legend.get_title():
            legend.get_title().set_fontsize(legend_title_fontsize)

    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def prepare_graph_from_excel(network_path: Path) -> nx.Graph:
    """Reads excel file describing the pandapower network and creates a nx.Graph from it.

    Args:
    -----
        network_path: Path to the network's Excel file.

    Returns:
    --------
        A networkx graph representation of the pandapower network.
    """
    bus_df = pd.read_excel(network_path, sheet_name="bus")
    line_df = pd.read_excel(network_path, sheet_name="line")
    trafo_df = pd.read_excel(network_path, sheet_name="trafo")

    G = nx.Graph()

    for _, bus in bus_df.iterrows():
        G.add_node(bus['name'])

    for _, line in line_df.iterrows():
        from_bus_name = bus_df.loc[line['from_bus']]['name']
        to_bus_name = bus_df.loc[line['to_bus']]['name']
        G.add_edge(from_bus_name, to_bus_name, name=line['name'], type='line')

    for _, trafo in trafo_df.iterrows():
        hv_bus_name = bus_df.loc[trafo['hv_bus']]['name']
        lv_bus_name = bus_df.loc[trafo['lv_bus']]['name']
        G.add_edge(hv_bus_name, lv_bus_name, name=trafo['name'], type='transformer')

    return G


def calculate_violation_metrics(
    res_bus: pd.DataFrame, res_line: pd.DataFrame, res_trafo: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes time-series results to get violation frequency and severity.

    Args:
    -----
        res_bus: DataFrame containing the voltage profile timeseries at each bus.
        res_line: DataFrame containing the line loadings timeseries of each line.
        res_trafo: DataFrame containing the transformer loadings timeseries of each transformer.

    Returns:
    --------
        bus_metrics: DataFrame containing violation metrics for each bus.
        all_edge_metrics: DataFrame containing violation metrics for all lines and transformers.
    """
    # Process Buses
    # Identify "significant" violations that are beyond the noise threshold.
    significant_voltage_violations = (res_bus <= UNDER_VOLTAGE_SEVERE) | (res_bus >= OVER_VOLTAGE_SEVERE)
    bus_metrics = pd.DataFrame(index=res_bus.columns)
    bus_metrics['freq'] = significant_voltage_violations.mean()

    # Calculate deviation from the original start threshold.
    deviation = np.maximum(UNDER_VOLTAGE_START - res_bus, res_bus - OVER_VOLTAGE_START).clip(lower=0)
    # Calculate mean severity only for the significant violations.
    bus_metrics['severity'] = deviation[significant_voltage_violations].mean()
    bus_metrics.fillna(0, inplace=True)

    # Process Lines
    line_metrics = pd.DataFrame(index=res_line.columns)
    if not res_line.empty:
        significant_line_overloads = res_line > OVERLOAD_SEVERE
        line_metrics['freq'] = significant_line_overloads.mean()
        # Calculate overload deviation from the original start threshold.
        overload = (res_line - OVERLOAD_START).clip(lower=0)
        # Calculate mean severity only for the significant overloads.
        line_metrics['severity'] = overload[significant_line_overloads].mean()
        line_metrics.fillna(0, inplace=True)

    # Process Transformers
    trafo_metrics = pd.DataFrame(index=res_trafo.columns)
    if not res_trafo.empty:
        significant_trafo_overloads = res_trafo > OVERLOAD_SEVERE
        trafo_metrics['freq'] = significant_trafo_overloads.mean()
        # Calculate overload deviation from the original start threshold.
        overload = (res_trafo - OVERLOAD_START).clip(lower=0)
        # Calculate mean severity only for the significant overloads.
        trafo_metrics['severity'] = overload[significant_trafo_overloads].mean()
        trafo_metrics.fillna(0, inplace=True)

    # Combine all edges into a single DataFrame
    all_edge_metrics = pd.concat([line_metrics, trafo_metrics])

    return bus_metrics, all_edge_metrics


def get_custom_network_coordinates() -> dict:
    """Generates custom (x, y) network coordinates that define its layout for plotting.

    Returns:
    --------
        A dictionary where keys are bus names and values are (x, y) tuples.
    """
    pos = {}

    # Define the vertical (y) levels for each voltage tier
    y_380kv = 10.0
    y_150kv_a = 9.3
    y_150kv_b = 8.5
    y_25kv_a = 7.5
    y_25kv_b = 6.5
    y_23kv_a = 6.0
    y_23kv_b = y_23kv_a
    y_10kv_a = 5.2
    y_10kv_b = 4.5

    # Define the horizontal (x) positions for the main branches
    x_center = 5.0
    x_main_left = 1.5
    x_main_right = 8.5

    # 380kV level
    pos['b_380_grid'] = (x_center, y_380kv)

    # First 150kV level
    pos['b_150_Waterningen'] = (x_main_left + 1.0, y_150kv_a)
    pos['b_150_Krimpen'] = (x_main_right, y_150kv_a)

    # Second 150kV level
    # Left Branch
    pos['b_150_Rijswijk'] = (x_main_left - 1.0, y_150kv_b)
    pos['b_150_sGravenhage'] = (x_main_left + 1.3, y_150kv_b)
    pos['b_150_Voorburg'] = (x_main_left + 4.0, y_150kv_b)
    # Right Branch
    pos['b_150_Ommoord'] = (x_main_right - 0.8, y_150kv_b)
    pos['b_150_Zuidwijk'] = (x_main_right + 2.2, y_150kv_b)

    # 25kV levels
    # First 25kV tier
    pos['b_25_HVS_Zuid'] = (pos['b_150_Rijswijk'][0], y_25kv_a)
    pos['b_25_HVS_Centrale'] = (pos['b_150_sGravenhage'][0], y_25kv_a)
    pos['b_25_HVS_Oost'] = (pos['b_150_Voorburg'][0], y_25kv_a)
    pos['b_25_RoCa'] = (pos['b_150_Ommoord'][0] - 0.5, y_25kv_a)
    pos['b_25_Ommoord'] = (pos['b_150_Ommoord'][0] + 1.0, y_25kv_a)

    # Second 25kV tier
    pos['b_25_Laagveen'] = (pos['b_25_HVS_Zuid'][0], y_25kv_b)
    pos['b_25_DH_EB'] = (pos['b_25_HVS_Centrale'][0] - 1.0, y_25kv_b)
    pos['b_25_DH'] = (pos['b_25_HVS_Centrale'][0] + 0.5, y_25kv_b)
    pos['b_25_DH_CHP'] = (pos['b_25_HVS_Centrale'][0] + 1.5, y_25kv_b)
    pos['b_25_Nootdorp2'] = (pos['b_25_HVS_Oost'][0], y_25kv_b)
    pos['b_25_Noordsingel'] = (pos['b_25_HVS_Oost'][0] + 1.5, y_25kv_b)
    pos['b_25_Bleiswijk'] = (pos['b_25_Ommoord'][0], y_25kv_b)

    # 23kV level
    pos['b_23_Nootdorp2'] = (pos['b_25_Nootdorp2'][0] - 0.5, y_23kv_a)
    pos['b_23_Zuidwijk'] = (pos['b_150_Zuidwijk'][0] - 1.2, y_23kv_a)
    pos['b_23_RDAM'] = (pos['b_150_Zuidwijk'][0], y_23kv_b)

    # First 10kV level
    pos['b_10_Laagveen'] = (pos['b_25_Laagveen'][0], y_10kv_a)
    pos['b_10_HVS_Centrale'] = (pos['b_25_HVS_Centrale'][0] - 0.8, y_10kv_a)
    pos['b_10_Nootdorp2'] = (pos['b_23_Nootdorp2'][0] - 1.5, y_10kv_a)
    pos['b_10_Noordsingel'] = (pos['b_25_Noordsingel'][0] - 2.0, y_10kv_a)
    pos['b_10_Ommoord'] = (pos['b_25_Ommoord'][0] - 2.1, y_10kv_a)
    pos['b_10_Bleiswijk'] = (pos['b_25_Bleiswijk'][0] - 0.5, y_10kv_a)

    # Second 10kV level
    pos['b_10_DH_1'] = (pos['b_10_Laagveen'][0], y_10kv_b)
    pos['b_10_DH_2'] = (pos['b_10_HVS_Centrale'][0], y_10kv_b)
    pos['b_10_DH_3'] = (pos['b_10_Nootdorp2'][0], y_10kv_b)
    pos['b_10_DH_4'] = (pos['b_10_Noordsingel'][0], y_10kv_b)
    pos['b_10_RDAM'] = (pos['b_10_Ommoord'][0], y_10kv_b)
    pos['b_10_Tuinders'] = (pos['b_10_Bleiswijk'][0], y_10kv_b)

    return pos


def sort_result_df_columns_by_voltage_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the columns of a DataFrame based on voltage levels indicated in the column names."""
    def get_voltage_from_name(col_name: str) -> tuple[int, ...]:
        matches = re.findall(r'(\d+)', col_name)
        return tuple(map(int, matches)) if matches else (0,)

    sorted_columns = sorted(df.columns.tolist(), key=get_voltage_from_name, reverse=True)
    return df[sorted_columns]


def generate_powerflow_statistics(
    experiment_path: Path,
    thermal_limit_percent: float = OVERLOAD_START,
    upper_voltage_limit: float = OVER_VOLTAGE_START,
    lower_voltage_limit: float = UNDER_VOLTAGE_START,
    sustained_violation_streak_hours: int = SUSTAINED_VIOLATION_STREAK_HOURS,
) -> None:
    """Analyzes power flow results and generates a comprehensive statistics CSV file.

    This function computes a finalized set of metrics for voltage and thermal performance, including
    basic statistics, violation frequencies, percentiles, and the number of sustained violation events.
    """
    paths = {
        'bus': experiment_path / "res_bus/vm_pu_with_names.csv",
        'line': experiment_path / "res_line/loading_percent_with_names.csv",
        'trafo': experiment_path / "res_trafo/loading_percent_with_names.csv"
    }

    df_bus = pd.read_csv(paths['bus'], index_col='snapshot', parse_dates=True)
    df_line = pd.read_csv(paths['line'], index_col='snapshot', parse_dates=True)
    df_trafo = pd.read_csv(paths['trafo'], index_col='snapshot', parse_dates=True)

    total_timesteps = len(df_bus)

    def analyze_thermal_component(df: pd.DataFrame, component_type: str) -> tuple[list[dict], list[dict]]:
        """Helper to process line or transformer loading data."""
        all_component_stats = []
        num_components = len(df.columns)

        for name in df.columns:
            series = df[name]
            is_overloaded = series > thermal_limit_percent
            overload_timesteps = is_overloaded.sum()

            num_sustained_events = 0
            if overload_timesteps > 0:
                grouper = (is_overloaded != is_overloaded.shift()).cumsum()
                streaks = is_overloaded.groupby(grouper).transform('size')
                overload_streaks = streaks[is_overloaded]
                if not overload_streaks.empty:
                    unique_streaks = overload_streaks.groupby(grouper[is_overloaded]).first()
                    num_sustained_events = (unique_streaks > SUSTAINED_VIOLATION_STREAK_HOURS).sum()

            stats_for_one_component = [
                {'Metric': 'Min Loading (%)', 'Value': series.min()},
                {'Metric': 'Max Loading (%)', 'Value': series.max()},
                {'Metric': 'Mean Loading (%)', 'Value': series.mean()},
                {'Metric': 'Median Loading (%)', 'Value': series.median()},
                {'Metric': 'Overload Frequency (%)', 'Value': (overload_timesteps / total_timesteps) * 100},
                {'Metric': '90th Percentile Loading (%)', 'Value': series.quantile(0.90)},
                {'Metric': 'Total Overload Hours', 'Value': overload_timesteps},
                {
                    'Metric': f'Sustained Overload Events (>{SUSTAINED_VIOLATION_STREAK_HOURS}h)',
                    'Value': num_sustained_events,
                },
            ]
            for stat in stats_for_one_component:
                stat.update({'Component_Type': component_type, 'Component_Name': name})
            all_component_stats.extend(stats_for_one_component)

        summary_stats = []
        df_stats = pd.DataFrame(all_component_stats)

        def get_summary(metric: str, func: str, title: str) -> dict:
            target_series = df_stats[df_stats['Metric'] == metric].set_index('Component_Name')['Value']
            if not target_series.empty and target_series.sum() > 0:
                component_name = getattr(target_series, func)()
                value = target_series.get(component_name, 0)
                return {'Metric': f'{component_type} {title}', 'Component_Name': component_name, 'Value': value}

            # Return a default dictionary to ensure the row always exists
            return {'Metric': f'{component_type} {title}', 'Component_Name': 'N/A', 'Value': 0}

        summary_stats.append(
            get_summary(metric='Max Loading (%)', func='idxmax', title='with Highest Peak Load')
        )
        summary_stats.append(
            get_summary(
                metric=f'Sustained Overload Events (>{SUSTAINED_VIOLATION_STREAK_HOURS}h)',
                func='idxmax',
                title='with Most Sustained Events'
            )
        )

        total_overloads = df_stats[df_stats['Metric'] == 'Total Overload Hours']['Value'].sum()
        pct_overloaded = (df > thermal_limit_percent).any().sum() / num_components * 100 if num_components > 0 else 0

        sustained_metric_name = f'Sustained Overload Events (>{SUSTAINED_VIOLATION_STREAK_HOURS}h)'
        sustained_events = df_stats[df_stats['Metric'] == sustained_metric_name]['Value']
        pct_with_sustained = (sustained_events > 0).sum() / num_components * 100 if num_components > 0 else 0

        summary_stats.append(
            {'Metric': f'Total Overload Hours ({component_type.lower()}-hours)', 'Value': total_overloads}
        )
        summary_stats.append({'Metric': f'% of {component_type}s that Ever Overload', 'Value': pct_overloaded})
        summary_stats.append({'Metric': f'% of {component_type}s with Sustained Events', 'Value': pct_with_sustained})

        # Network-wide aggregate stats
        if not df.empty:
            all_values = df.values.flatten()
            summary_stats.append(
                {'Metric': f'Mean Network Loading (%) ({component_type.lower()})', 'Value': np.mean(all_values)}
            )
            summary_stats.append(
                {'Metric': f'Median Network Loading (%) ({component_type.lower()})', 'Value': np.median(all_values)}
            )
            summary_stats.append(
                {
                    'Metric': f'90th Percentile Network Loading (%) ({component_type.lower()})',
                    'Value': np.quantile(all_values, 0.9)
                }
            )

        for stat in summary_stats:
            stat.update({'Component_Type': 'Network Summary', 'Component_Name': stat.get('Component_Name', 'All')})

        return summary_stats, all_component_stats


    def analyze_voltage_component(df: pd.DataFrame, component_type: str) -> tuple[list[dict], list[dict]]:
        """Helper to process bus voltage data."""
        all_component_stats = []
        df_clean = df.drop(columns=['b_380_grid'], errors='ignore')
        num_buses = len(df_clean.columns)

        for name in df_clean.columns:
            series = df_clean[name]
            is_violating = (series < lower_voltage_limit) | (series > upper_voltage_limit)
            violation_timesteps = is_violating.sum()

            num_sustained_events = 0
            if violation_timesteps > 0:
                grouper = (is_violating != is_violating.shift()).cumsum()
                streaks = is_violating.groupby(grouper).transform('size')
                violation_streaks = streaks[is_violating]
                if not violation_streaks.empty:
                    unique_streaks = violation_streaks.groupby(grouper[is_violating]).first()
                    num_sustained_events = (unique_streaks > sustained_violation_streak_hours).sum()

            undervoltage_hours = (series < lower_voltage_limit).sum()
            overvoltage_hours = (series > upper_voltage_limit).sum()

            stats_for_one_component = [
                {'Metric': 'Min Voltage (p.u.)', 'Value': series.min()},
                {'Metric': 'Max Voltage (p.u.)', 'Value': series.max()},
                {'Metric': 'Mean Voltage (p.u.)', 'Value': series.mean()},
                {'Metric': 'Median Voltage (p.u.)', 'Value': series.median()},
                {'Metric': 'Violation Frequency (%)', 'Value': (violation_timesteps / total_timesteps) * 100},
                {'Metric': '10th Percentile Voltage (p.u.)', 'Value': series.quantile(0.10)},
                {'Metric': '90th Percentile Voltage (p.u.)', 'Value': series.quantile(0.90)},
                {'Metric': 'Total Violation Hours', 'Value': violation_timesteps},
                {
                    'Metric': f'Sustained Violation Events (>{sustained_violation_streak_hours}h)',
                    'Value': num_sustained_events
                },
                {'Metric': 'Number of Undervoltage Hours', 'Value': undervoltage_hours},
                {'Metric': 'Undervoltage Frequency (%)', 'Value': (undervoltage_hours / total_timesteps) * 100},
                {'Metric': 'Number of Overvoltage Hours', 'Value': overvoltage_hours},
                {'Metric': 'Overvoltage Frequency (%)', 'Value': (overvoltage_hours / total_timesteps) * 100},
            ]
            for stat in stats_for_one_component:
                stat.update({'Component_Type': component_type, 'Component_Name': name})
            all_component_stats.extend(stats_for_one_component)

        summary_stats = []
        df_stats = pd.DataFrame(all_component_stats)

        def get_summary(metric: str, func: str, title: str) -> dict:
            target_series = df_stats[df_stats['Metric'] == metric].set_index('Component_Name')['Value']
            if not target_series.empty and target_series.sum() > 0:
                component_name = getattr(target_series, func)()
                value = target_series.get(component_name, 0)
                return {'Metric': f'Bus {title}', 'Component_Name': component_name, 'Value': value}
            return {'Metric': f'Bus {title}', 'Component_Name': 'N/A', 'Value': 0}

        summary_stats.append(
            get_summary(metric='Total Violation Hours', func='idxmax', title='with Most Violation Hours')
        )
        summary_stats.append(
            get_summary(
                metric=f'Sustained Violation Events (>{sustained_violation_streak_hours}h)',
                func='idxmax',
                title='with Most Sustained Events'
            )
        )
        summary_stats.append(
            get_summary(metric='Number of Undervoltage Hours', func='idxmax', title='with Most Undervoltage Hours')
        )
        summary_stats.append(
            get_summary(metric='Number of Overvoltage Hours', func='idxmax', title='with Most Overvoltage Hours')
        )

        is_violating_globally = (df_clean < lower_voltage_limit) | (df_clean > upper_voltage_limit)
        pct_violating = is_violating_globally.any().sum() / num_buses * 100 if num_buses > 0 else 0

        sustained_metric_name = f'Sustained Violation Events (>{sustained_violation_streak_hours}h)'
        sustained_events = df_stats[df_stats['Metric'] == sustained_metric_name]['Value']
        pct_with_sustained = (sustained_events > 0).sum() / num_buses * 100 if num_buses > 0 else 0

        vol_over = df_stats[df_stats['Metric'] == 'Number of Overvoltage Hours']['Value'].sum()
        vol_under = df_stats[df_stats['Metric'] == 'Number of Undervoltage Hours']['Value'].sum()

        summary_stats.append({'Metric': 'Volume of Overvoltage (bus-hours)', 'Value': vol_over})
        summary_stats.append({'Metric': 'Volume of Undervoltage (bus-hours)', 'Value': vol_under})

        summary_stats.append({'Metric': '% of Buses with any Violation', 'Value': pct_violating})
        summary_stats.append({'Metric': '% of Buses with Sustained Events', 'Value': pct_with_sustained})
        summary_stats.append({'Metric': 'Max System Voltage (p.u.)', 'Value': df.max().max()})
        summary_stats.append({'Metric': 'Min System Voltage (p.u.)', 'Value': df.min().min()})

        # Network-wide aggregate stats
        if not df_clean.empty:
            all_values = df_clean.values.flatten()
            summary_stats.append({'Metric': 'Mean Network Voltage (p.u.)', 'Value': np.mean(all_values)})
            summary_stats.append({'Metric': 'Median Network Voltage (p.u.)', 'Value': np.median(all_values)})
            summary_stats.append(
                {'Metric': '10th Percentile Network Voltage (p.u.)', 'Value': np.quantile(all_values, 0.1)}
            )
            summary_stats.append(
                {'Metric': '90th Percentile Network Voltage (p.u.)', 'Value': np.quantile(all_values, 0.9)}
            )

        for stat in summary_stats:
            stat.update({'Component_Type': 'Network Summary', 'Component_Name': stat.get('Component_Name', 'All')})

        return summary_stats, all_component_stats

    # Execute main analysis
    all_stats = []

    line_summary, line_details = analyze_thermal_component(df_line, 'Line')
    trafo_summary, trafo_details = analyze_thermal_component(df_trafo, 'Transformer')
    bus_summary, bus_details = analyze_voltage_component(df_bus, 'Bus')

    all_stats.extend(line_summary)
    all_stats.extend(trafo_summary)
    all_stats.extend(bus_summary)
    all_stats.extend(line_details)
    all_stats.extend(trafo_details)
    all_stats.extend(bus_details)

    final_df = pd.DataFrame(all_stats)
    final_df = final_df[['Component_Type', 'Component_Name', 'Metric', 'Value']]
    final_df['Value'] = final_df['Value'].round(4)

    output_path = experiment_path / "statistics.csv"
    final_df.to_csv(output_path, index=False)
