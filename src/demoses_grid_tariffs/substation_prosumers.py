import pandas as pd
import pyomo.environ as pyo
import yaml

from pathlib import Path
from cronian.base_model import create_optimization_model
from cronian.prosumers import build_prosumer_model

from demoses_grid_tariffs.tariffs import Tariffs


def create_prosumer_optimization_model(
    prosumer: dict,
    electricity_price: pd.DataFrame,
    timeseries_data: pd.DataFrame,
    tariffs: Tariffs | None = None,
) -> pyo.ConcreteModel:
    """Create prosumer optimization model.

    Args:
    -----
        prosumer: prosumer cronian configuration.
        electricity_price: DataFrame containing the electricity price time series.
        timeseries_data: DataFrame containing the time series data for the prosumer (e.g. load and generation profiles).
        tariffs: Grid tariffs (capacity and/or volumetric, on withdrawal and/or injection) to apply.
            If None, no grid tariffs are applied. See demoses_grid_tariffs.tariffs.Tariffs.

    Returns:
    --------
        A Pyomo ConcreteModel representing the prosumer optimization problem.
    """
    tariffs = tariffs or Tariffs()

    prosumer_id = prosumer["id"]
    num_timesteps = len(electricity_price)
    model = create_optimization_model(None, electricity_price, num_timesteps)  # Though we don't use elec price here
    model.name = f"Optimization model of prosumer--{prosumer_id}"
    model = build_prosumer_model(model, prosumer, timeseries_data, num_timesteps, storage_model="complex")
    model = model.create_instance()

    # Power split: split electric power into non-negative injection and withdrawal variables.
    # This allows tariffs to be applied separately to the power withdrawn from and injected into the grid.
    model.power_injection = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    model.power_withdrawal = pyo.Var(model.time, domain=pyo.NonNegativeReals)

    # Add constraint to relate the split variables back with electric power variable
    def electric_power_split_rule(model, t):
        return (
            getattr(model, f"{prosumer_id}_electric_power")[t]
            == model.power_injection[t] - model.power_withdrawal[t]
        )
    model.power_split = pyo.Constraint(model.time, rule=electric_power_split_rule)

    # Tariffs: add params, variables, and constraints for each configured direction.
    model = add_tariff_components(model, tariffs, snapshots=electricity_price.index)

    # Objective
    model = add_objective_function_to_model(prosumer, model, electricity_price, tariffs)

    return model


def get_direction_power_var(model: pyo.ConcreteModel, direction: str) -> pyo.Var:
    """Return the non-negative power variable for a tariff direction."""
    if direction == "withdrawal":
        return model.power_withdrawal
    if direction == "injection":
        return model.power_injection
    raise ValueError(f"Unknown tariff direction: {direction}. Use 'withdrawal' or 'injection'.")


def add_tariff_components(model: pyo.ConcreteModel, tariffs: Tariffs, snapshots: pd.DatetimeIndex) -> pyo.ConcreteModel:
    """Add volumetric and/or capacity tariff components for withdrawal and/or injection."""
    for direction in ("withdrawal", "injection"):
        direction_tariff = getattr(tariffs, direction)
        if direction_tariff.has_volumetric:
            model = add_volumetric_tariff(model, direction_tariff.volumetric, direction)
        if direction_tariff.has_capacity:
            model = add_capacity_tariff(
                model, snapshots, direction_tariff.cap_level, direction_tariff.cap_weights, direction
            )
    return model


def add_objective_function_to_model(
    prosumer: dict,
    model: pyo.ConcreteModel,
    electricity_price: pd.DataFrame,
    tariffs: Tariffs,
) -> pyo.ConcreteModel:
    """Add a cost-minimizing and revenue-maximizing objective to the Pyomo model.

    Args:
    -----
        prosumer: prosumer cronian configuration.
        model: The Pyomo model to which the objective function will be added.
        electricity_price: DataFrame containing the electricity price time series.
        tariffs: The grid tariffs whose cost terms are added to the objective.

    Returns:
    --------
        The Pyomo model with the objective function added.
    """
    prosumer_id = prosumer["id"]

    model.e_price = pyo.Param(
        model.time,
        initialize={t: p for t, p in zip(model.time, electricity_price["electricity_price"].values)},
    )

    def prosumer_cost_rule(model):
        total_cost = -1 * sum(
            model.e_price[t] * getattr(model, f"{prosumer_id}_electric_power")[t] for t in model.time
        )

        for direction in ("withdrawal", "injection"):
            direction_tariff = getattr(tariffs, direction)
            if direction_tariff.has_volumetric:
                total_cost += get_volumetric_tariff_cost(model, direction)
            if direction_tariff.has_capacity:
                total_cost += get_capacity_tariff_cost(model, direction)

        return total_cost

    model.objective = pyo.Objective(rule=prosumer_cost_rule, sense=pyo.minimize)

    return model


def add_volumetric_tariff(model: pyo.ConcreteModel, vol_tariff: pd.Series, direction: str) -> pyo.ConcreteModel:
    """Add a snapshot-indexed volumetric tariff parameter [€/MWh] for a direction."""
    setattr(
        model,
        f"vol_tariff_{direction}",
        pyo.Param(model.time, initialize={t: float(v) for t, v in zip(model.time, vol_tariff.values)}),
    )
    return model


def get_volumetric_tariff_cost(model: pyo.ConcreteModel, direction: str) -> pyo.Expression:
    """Volumetric tariff cost = sum over time of tariff[t] * power[t] for the direction."""
    power_var = get_direction_power_var(model, direction)
    vol_param = getattr(model, f"vol_tariff_{direction}")
    return sum(vol_param[t] * power_var[t] for t in model.time)


def add_capacity_tariff(
    model: pyo.ConcreteModel,
    snapshots: pd.DatetimeIndex,
    cap_level: float,
    cap_weights: pd.Series | None,
    direction: str,
) -> pyo.ConcreteModel:
    """Add a weighted-kWmax capacity tariff structure for a direction.

    Each snapshot's power is multiplied by the (optional) hourly weight, then the monthly
    maximum of these weighted values is charged at ``cap_level`` (weighted kW max). If
    ``cap_weights`` is None, the tariff is unweighted (all weights equal to one).
    """
    power_var = get_direction_power_var(model, direction)
    months = snapshots.to_series().dt.to_period("M")

    # Shared month set and time->month mapping (added once, reused across directions).
    if not hasattr(model, "tariff_months"):
        model.tariff_months = pyo.Set(initialize=[str(m) for m in months.unique()])
        month_map = {t: str(m) for t, m in zip(model.time, months)}
        model.time_to_month = pyo.Param(model.time, initialize=month_map, within=pyo.Any)

    # Hourly weight parameter (defaults to ones when no weights are provided).
    if cap_weights is not None:
        weight_values = [float(w) for w in cap_weights.values]
    else:
        weight_values = [1.0 for _ in model.time]
    setattr(
        model,
        f"cap_weight_{direction}",
        pyo.Param(model.time, initialize={t: w for t, w in zip(model.time, weight_values)}),
    )

    # Capacity tariff level [€/MW-month] for this direction.
    setattr(model, f"cap_level_{direction}", pyo.Param(initialize=float(cap_level), mutable=False))

    # Monthly maximum of the weighted power for this direction.
    max_var = pyo.Var(model.tariff_months, domain=pyo.NonNegativeReals)
    setattr(model, f"max_{direction}", max_var)

    weight_param = getattr(model, f"cap_weight_{direction}")

    def max_rule(m, t):
        return max_var[m.time_to_month[t]] >= weight_param[t] * power_var[t]

    setattr(model, f"max_{direction}_constraint", pyo.Constraint(model.time, rule=max_rule))

    return model


def get_capacity_tariff_cost(model: pyo.ConcreteModel, direction: str) -> pyo.Expression:
    """Capacity tariff cost = level * sum over months of the monthly weighted maximum."""
    max_var = getattr(model, f"max_{direction}")
    cap_level = getattr(model, f"cap_level_{direction}")
    return sum(cap_level * max_var[m] for m in model.tariff_months)


def generate_prosumer_cronian_config(
    splitted_substation_profiles: pd.DataFrame, column_name: str, flex_factor: float = 0.2, output: Path = None
) -> dict:
    """
    Generates a Cronian configuration for the specified (column_name) substation prosumer.

    The generated yaml looks like this:

    Prosumers:
        name: <column_name>
        id: P01_<column_name>
        demand:
            sub_stattion_demand:
                carrier: some_carrier
                base:
                    n_profile: <column_name_load>
                    peak: <max_value_of_column_name_load>
        assets:
            generic_asset:
                behavior_type: converter
                input: electricity
                output: some_carrier
                installed_capacity: <specified_value>
                efficiency: 0.99
            battery:
                behavior_type: storage
                input: electricity
                output: electricity
                energy_capacity: <specified_value>
                initial_energy: 0
                charge_capacity: <specified_value>
                discharge_capacity: <specified_value>
                charge_efficiency: 0.95
                discharge_efficiency: 0.95
            solar_pv:
                behavior_type: generator
                input: light
                output: electricity
                installed_capacity: 1
                availability_factor: <column_name_sgen>
                operational_costs:
                    marginal_cost_linear: 0.0
                    marginal_cost_quadratic: 0.0

    Args:
    -----
        splitted_substation_profiles: DataFrame containing the split profiles of
        substations (column_name_load and column_name_sgen).
        column_name: The base name of the columns in the DataFrame that contain
            the load and generation profiles.
        flex_factor: A factor that determines the size of the battery storage
            relative to the load profile.
        output: Optional path to save the generated YAML string as a file.

    Returns:
    --------
        A dictionary representing the Cronian configuration for the prosumer.
    """
    df = splitted_substation_profiles.copy()
    load_col = f"{column_name}_load"
    sgen_col = f"{column_name}_sgen"

    max_load = df[load_col].max()
    max_sgen = df[sgen_col].max()
    sum_load = df[load_col].sum()

    generic_asset_cap = float(f"{max_load * 1.1:.0f}") if max_load > 5.0 else 5.0
    battery_energy_cap = float(f"{sum_load * 0.05:.0f}") if sum_load > 500.0 else 500.0
    battery_power_cap = (
        float(f"{max_sgen + max_load * flex_factor:.0f}") if max_sgen + max_load * flex_factor > 10.0 else 10.0
    )

    config = {
        "Prosumers": {
            "name": column_name,
            "id": f"P0_{column_name}",
            "demand": {
                "sub_station_demand": {
                    "carrier": "some_carrier",
                    "base": {
                        "n_profile": load_col,
                        "peak": 1
                    }
                }
            },
            "assets": {
                "generic_asset": {
                    "behavior_type": "converter",
                    "input": "electricity",
                    "output": "some_carrier",
                    "installed_capacity": generic_asset_cap,
                    "efficiency": 0.99
                },
                "battery": {
                    "behavior_type": "storage",
                    "input": "electricity",
                    "output": "electricity",
                    "energy_capacity": battery_energy_cap,
                    "initial_energy": 0,
                    "charge_capacity": battery_power_cap,
                    "discharge_capacity": battery_power_cap,
                    "charge_efficiency": 0.95,
                    "discharge_efficiency": 0.95
                },
                "solar_pv": {
                    "behavior_type": "generator",
                    "input": "light",
                    "output": "electricity",
                    "installed_capacity": 1,
                    "availability_factor": sgen_col,
                    "operational_costs": {
                        "marginal_cost_linear": 0.0,
                        "marginal_cost_quadratic": 0.0
                    }
                }
            }
        }
    }

    if output:
        output_path = output / f"P0_{column_name}.yaml"
        with open(output_path, "w") as f:
            # Set sort_keys=False to keep the order in the output file
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    return config
