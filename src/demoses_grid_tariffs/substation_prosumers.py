import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
from cronian.base_model import create_optimization_model
from cronian.prosumers import build_prosumer_model


def create_prosumer_optimization_model(
    prosumer: dict,
    electricity_price: pd.DataFrame,
    timeseries_data: pd.DataFrame,
    vol_tou_tariffs_demand: pd.DataFrame | None = None,
    cap_tariff: float | None = None,
    cap_tariff_weights_monthly: pd.DataFrame | None = None,
) -> pyo.ConcreteModel:
    """Create prosumer optimization model.
        
    Args:
    -----
        prosumer: prosumer cronian configuration.
        electricity_price: DataFrame containing the electricity price time series.
        timeseries_data: DataFrame containing the time series data for the prosumer (e.g. load and generation profiles).
        vol_tou_tariffs_demand: DataFrame containing the volumetric time-of-use tariffs on demand.
        cap_tariff: Capacity tariff [€/MW].
        cap_tariff_weights_monthly: DataFrame containing monthly weights for the capacity tariff.

    Returns:
    --------
        A Pyomo ConcreteModel representing the prosumer optimization problem.

    Raises:
    -------
        ValueError: If only one of cap_tariff or cap_tariff_weights_monthly is provided without the other.
    """
    if (cap_tariff is None) != (cap_tariff_weights_monthly is None):
        raise ValueError("Both cap_tariff and cap_tariff_weights_monthly must be provided together.")
    
    prosumer_id = prosumer["id"]
    num_timesteps = len(electricity_price)
    model = create_optimization_model(None, electricity_price, num_timesteps) # Though we don't use elec price here
    model.name = f"Optimization model of prosumer--{prosumer_id}"
    model = build_prosumer_model(model, prosumer, timeseries_data, num_timesteps, storage_model="complex")
    model = model.create_instance()

    # Power split: add constraint to split electric power into injection and withdrawal variables.
    # This allows us to apply the vol_tou_tariffs_demand only to the power withdrawn from the grid.
    model.power_injection = pyo.Var(model.time, domain=pyo.NonNegativeReals)
    model.power_withdrawal = pyo.Var(model.time, domain=pyo.NonNegativeReals)

    # Add constraint to relate the split variables back with electric power variable
    def electric_power_split_rule(model, t):
        return (
            getattr(model, f"{prosumer_id}_electric_power")[t]
            == model.power_injection[t] - model.power_withdrawal[t]
        )
    model.power_split = pyo.Constraint(model.time, rule=electric_power_split_rule)

    # Tariffs: add params, variables, and constraints needed to model the application of tariffs.
    if vol_tou_tariffs_demand is not None:
        model = add_volumetric_tariff(model, vol_tou_tariffs_demand)

    if cap_tariff is not None and cap_tariff_weights_monthly is not None:
        model = add_capacity_tariff(
            model,
            snapshots=electricity_price.index,
            cap_tariff=cap_tariff,
            cap_tariff_weights_monthly=cap_tariff_weights_monthly,
        )

    # Objective
    model = add_objective_function_to_model(
        prosumer,
        model,
        electricity_price,
        vol_tou_tariffs_demand=vol_tou_tariffs_demand,
        cap_tariff=cap_tariff,
        cap_tariff_weights_monthly=cap_tariff_weights_monthly,
    )

    return model


def add_objective_function_to_model(
    prosumer: dict,
    model: pyo.ConcreteModel,
    electricity_price: pd.DataFrame,
    vol_tou_tariffs_demand: pd.DataFrame | None = None,
    cap_tariff: float | None = None,
    cap_tariff_weights_monthly: pd.DataFrame | None = None,
) -> pyo.ConcreteModel:
    """Add a cost-minimizing and revenue-maximizing objective to the Pyomo model.

    Args:
    -----
        prosumer: prosumer cronian configuration.
        model: The Pyomo model to which the objective function will be added.
        electricity_price: DataFrame containing the electricity price time series.
        vol_tou_tariffs_demand: DataFrame containing the volumetric time-of-use tariffs on demand [€/MWh].
        cap_tariff: Capacity tariff [€/MW].
        cap_tariff_weights_monthly: DataFrame containing monthly weights for the capacity tariff.

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
        electricity_cost = -1 * sum(
            model.e_price[t] * getattr(model, f"{prosumer_id}_electric_power")[t] for t in model.time
        )

        vol_tariff_cost = get_volumetric_tariff_cost(model) if vol_tou_tariffs_demand is not None else 0
        cap_tariff_cost = (
            get_capacity_tariff_cost(model) if cap_tariff is not None and cap_tariff_weights_monthly is not None
            else 0
        )

        return electricity_cost + vol_tariff_cost + cap_tariff_cost

    model.objective = pyo.Objective(rule=prosumer_cost_rule, sense=pyo.minimize)

    return model


def add_volumetric_tariff(model: pyo.ConcreteModel, vol_tou_tariffs_demand: pd.DataFrame | None) -> pyo.ConcreteModel:
    """Adds volumetric tariff parameters to the model."""
    if vol_tou_tariffs_demand is None:
        return model
    
    model.vol_tou_tariffs = pyo.Param(
        model.time,
        initialize={
            t: p for t, p in zip(model.time, vol_tou_tariffs_demand["vol_tou_tariffs"].values)
        },
    )

    return model


def get_volumetric_tariff_cost(model: pyo.ConcreteModel) -> pyo.Expression:
    return sum(model.vol_tou_tariffs[t] * model.power_withdrawal[t] for t in model.time)


def add_capacity_tariff(
    model: pyo.ConcreteModel,
    snapshots: pd.DatetimeIndex,
    cap_tariff: float | None,
    cap_tariff_weights_monthly: pd.DataFrame | None,
) -> pyo.ConcreteModel:
    """Adds monthly capacity tariff structure."""
    if cap_tariff is None or cap_tariff_weights_monthly is None:
        return model

    # Month mapping
    year = snapshots[0].year
    months = snapshots.to_series().dt.to_period("M")
    unique_months = months.unique()

    # Parameters and Sets
    model.months = pyo.Set(initialize=list(unique_months))
    model.cap_tariff = pyo.Param(initialize=cap_tariff, mutable=False)

    # Weights
    df = cap_tariff_weights_monthly.copy()
    df["month"] = pd.PeriodIndex(year=year, month=df["month"].astype(int), freq="M")
    df = df.set_index("month")
    weight_map = {m: df.loc[m, "value"] for m in unique_months}

    model.cap_tariff_weight = pyo.Param(model.months, initialize=weight_map)

    # Variable
    model.max_withdrawal = pyo.Var(model.months, domain=pyo.NonNegativeReals)

    # Constraint
    # Create a mapping of time steps to months for the capacity tariff application
    month_map = {t: m for t, m in zip(model.time, months)}
    model.time_to_month = pyo.Param(model.time, initialize=month_map, within=pyo.Any)

    def max_withdrawal_rule(model, t):
        return model.max_withdrawal[model.time_to_month[t]] >= model.power_withdrawal[t]
    model.max_withdrawal_constraint = pyo.Constraint(model.time, rule=max_withdrawal_rule)

    return model


def get_capacity_tariff_cost(model: pyo.ConcreteModel) -> pyo.Expression:
    return sum(
        model.cap_tariff * model.cap_tariff_weight[m] * model.max_withdrawal[m]
        for m in model.months
    )


def generate_prosumer_cronian_config(
    processed_substation_profiles: pd.DataFrame, column_name: str, flex_factor: float = 0.2, output: Path = None
) -> str:
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
                installed_capacity: <max_value_of_column_name_load * 1.02>
                efficiency: 1.0
            battery:
                behavior_type: storage
                input: electricity
                output: electricity
                energy_capacity: <sum_of_column_name_load * 0.05>
                initial_energy: 0
                charge_capacity: <max_of_column_name_sgen + max_of_column_name_load * flex_factor>
                discharge_capacity: <max_of_column_name_sgen + max_of_column_name_load * flex_factor>
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
        processed_substation_profiles: DataFrame containing the load
            (column_name_load) and generation (column_name_sgen) of substations.
        column_name: The base name of the columns in the DataFrame that contain
            the load and generation profiles.
        flex_factor: A factor that determines the size of the battery storage
            relative to the load profile.
        output: Optional path to save the generated YAML string as a file.
    """
    df = processed_substation_profiles.copy()
    load_col = f"{column_name}_load"
    sgen_col = f"{column_name}_sgen"

    max_load = df[load_col].max()
    max_sgen = df[sgen_col].max()
    sum_load = df[load_col].sum()

    yaml_str = f"""Prosumers:
  name: {column_name}
  id: P0_{column_name}
  demand:
    sub_station_demand:
      carrier: some_carrier
      base:
        n_profile: {load_col}
        peak: 1
  assets:
    generic_asset:
      behavior_type: converter
      input: electricity
      output: some_carrier
      installed_capacity: {max_load * 1.1:.0f}
      efficiency: 0.99
    battery:
      behavior_type: storage
      input: electricity
      output: electricity
      energy_capacity: {sum_load * 0.05:.0f}
      initial_energy: 0
      charge_capacity: {max_sgen + max_load * flex_factor:.0f}
      discharge_capacity: {max_sgen + max_load * flex_factor:.0f}
      charge_efficiency: 0.95
      discharge_efficiency: 0.95
    solar_pv:
      behavior_type: generator
      input: light
      output: electricity
      installed_capacity: 1
      availability_factor: {sgen_col}
      operational_costs:
        marginal_cost_linear: 0.0
        marginal_cost_quadratic: 0.0
    """
    # If output path is provided, write the yaml string to a file named {P0_{column_name}.yaml in the output directory
    if output:
        output_path = output / f"P0_{column_name}.yaml"
        with open(output_path, "w") as f:
            f.write(yaml_str)

    return yaml_str
