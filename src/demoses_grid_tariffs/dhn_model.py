from pathlib import Path

import logging
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from linopy.model import LinearExpression, Model

from demoses_grid_tariffs.helper_functions import (
    CARRIERS_ELEC_PROD_LINKS,
    calculate_heatpump_cop,
    get_assets_based_on_carrier_name,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Geothermal_SPF = 6.0  # Seasonal performance factor for geothermal systems
HT_ATES_SPF = 50.0  # Seasonal performance factor for HT-ATES systems
HT_ATES_FULL_LOAD_HOURS = 3000  # Full load hours for HT-ATES systems (10MW * 3000h = 30GWh energy capacity)


def optimize_district_heating_network(n: pypsa.Network, model: Model, solver_options: dict) -> pypsa.Network:
    """Run the operation and/or planning optimization of the district heating network.

    Args:
    -----
        n: PyPSA network model of the district heating network.
        model: Linopy model object of the PyPSA network.
        solver_options: Dictionary containing solver name and its parameters.

    Returns:
    --------
        The optimized PyPSA network object.
    """
    solver_name = list(solver_options.keys())[0]
    kwargs = solver_options[solver_name]
    model.solve(solver_name=solver_name, **kwargs)

    # model.print_infeasibilities()

    # Assign Linopy model solution back to the PyPSA network
    n.optimize.assign_solution()
    n.optimize.assign_duals()

    return n


def build_district_heating_network(
    csv_folder: Path,
    temperature: pd.DataFrame,
    heat_demand: pd.DataFrame,
    hydrogen_price: pd.DataFrame,
    electricity_price: pd.DataFrame,
    solar_availability: pd.DataFrame,
    static_prices: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    vol_tou_tariffs: pd.DataFrame | None = None,
) -> tuple[pypsa.Network, Model]:
    """Build the district heating network model.

    NOTE: It returns both the PyPSA network and the Linopy model so that the solution to the
        solved linopy model is assigned back to the PyPSA network for easy analysis and plotting.

    Args:
    -----
        csv_folder: Path to the folder containing CSV files describing the network.
        temperature: DataFrame containing temperature timeseries.
        heat_demand: DataFrame containing heat demand timeseries.
        hydrogen_price: DataFrame containing hydrogen price timeseries.
        electricity_price: DataFrame containing electricity price timeseries.
        solar_availability: DataFrame containing solar availability timeseries for solar thermal generators.
        static_prices: DataFrame with non-time varying prices for greengas, waste material, residual heat, etc.
        snapshots: Set of timestamps to consider in the optimization.
        vol_tou_tariffs: DataFrame containing volumetric TOU tariff in €/MWh (optional).

    Returns:
    --------
        tuple: The PyPSA network object and its underlying Linopy model.

    Raises:
    -------
        ValueError: If the length of any timeseries data is less than the number of snapshots.
    """
    # Check if the snapshots are valid
    for ts_data in [heat_demand, temperature, hydrogen_price, electricity_price, solar_availability]:
        if len(ts_data) < len(snapshots):
            raise ValueError(
                f"Data length mismatch: {len(ts_data)} < {len(snapshots)}. "
                "Ensure that the data covers all snapshots.",
            )

    n = create_network_from_csv(csv_folder=csv_folder, snapshots=snapshots)
    n = update_network_data(
        n=n,
        temperature=temperature,
        heat_demand=heat_demand,
        hydrogen_price=hydrogen_price,
        electricity_price=electricity_price,
        solar_availability=solar_availability,
        static_prices=static_prices,
        snapshots=snapshots,
    )

    if vol_tou_tariffs is not None:
        if len(vol_tou_tariffs) < len(snapshots):
            raise ValueError(
                f"Data length mismatch: {len(vol_tou_tariffs)} < {len(snapshots)}. "
                "Ensure that the volumetric TOU tariffs data covers all snapshots.",
            )
        logger.info("Adding volumetric TOU tariffs to the heat model...")
        n = add_volumetric_tou_tariffs(n, vol_tou_tariffs)
        logger.info("Successfully added volumetric TOU tariffs to the heat model.")

    # Create linopy model from the pypsa network
    model = n.optimize.create_model()

    # Add geothermal seasonal performance factor constraints
    model = add_geothermal_seasonal_performance_factor_constraints(n, model, SPF=Geothermal_SPF)

    # Add HT-ATES constraints
    model = add_ht_ates_constraints(n, model, SPF=HT_ATES_SPF, full_load_hours=HT_ATES_FULL_LOAD_HOURS)

    # Add electricity revenue from CHPs to the objective function
    chp_assets = []
    for carrier in CARRIERS_ELEC_PROD_LINKS:
        chp_assets.extend(get_assets_based_on_carrier_name(n, "Link", carrier))
    electricity_price_vals = electricity_price.loc[snapshots, "electricity_price"].values
    electricity_revenue_chps = build_electricity_revenue(n, model, chp_assets, electricity_price_vals)
    model = update_objective_function(model, electricity_revenue_chps)

    return n, model


def create_network_from_csv(csv_folder: Path, snapshots: pd.DatetimeIndex) -> pypsa.Network:
    """Create a pypsa network of the district heating model from CSV files.

    Args:
    -----
        csv_folder: Path to the folder containing CSV files describing the network.
        snapshots: Set of timestamps to consider in the optimization.

    Returns:
    --------
        The PyPSA network object loaded from CSV files.
    """
    n = pypsa.Network(name="District Heating Network", snapshots=snapshots)
    n.import_from_csv_folder(csv_folder)

    return n


def add_geothermal_seasonal_performance_factor_constraints(n: pypsa.Network, model: Model, SPF: float) -> Model:
    """Add seasonal performance factor (SPF) for geothermal systems.

    Geothermal unit/asset is modeled with a geothermal well (Generator), a booster heat pump (Link),
    and a geothermal link (Link) that connects the two.

    Initially, pypsa has already established the following constraint:
        Link-p1[Geothermal, t] == Generator-p[Geothermal, t] + COP_booster * Link-p0[booster, t]

    However, the above implies arbitrary combinations of heat from the geothermal well and booster
    heat pump. This new constraint ensures the share of heat from the geothermal well and booster
    heat pump is fixed according to the SPF as follows:
        Link-p1[Geothermal, t] / Link-p0[booster, t] == SPF

    Combining the two equations above gives:
        Generator-p[Geothermal, t] == (SPF - 1) * Link-p0[booster, t]

    Or in terms of heat only (the high-grade heat output of links is at bus1 (p1)):
        Generator-p[Geothermal, t] == (SPF - COP_booster) / COP_booster * Link-p1[booster, t]


    Args:
    -----
        n: PyPSA network object.
        model: The optimization model.
        SPF: The seasonal performance factor for geothermal systems.

    Returns:
    --------
        The updated optimization model with the new geothermal SPF constraints.
    """
    geo_links = get_assets_based_on_carrier_name(n, "Link", "Geothermal")
    geo_wells = get_assets_based_on_carrier_name(n, "Generator", "Geothermal well")
    booster_hps = get_assets_based_on_carrier_name(n, "Link", "Geothermal ASHP")

    if not (len(geo_links) == len(booster_hps) == len(geo_wells)):
        raise ValueError(
            "Geothermal is modeled with one geothermal well, one booster heat pump, and one geothermal link."
            f"Found mismatches: {len(geo_links)} geothermal links, {len(geo_wells)} geothermal wells, "
            f" and {len(booster_hps)} booster heat pumps."
        )

    new_cons = model.variables["Link-p"].sel(name=geo_links) == SPF * model.variables["Link-p"].sel(name=booster_hps)
    model.add_constraints(new_cons, name="Geothermal_SPF-constraint")

    return model


def add_ht_ates_constraints(
    n: pypsa.Network,
    model: Model,
    SPF: float,
    full_load_hours: float,
) -> Model:
    """Add HT-ATES seasonal performance factor constraints and energy and power capacities relation.

    For each HT-ATES system (charger, discharger, store), this enforces the following constraints:
        1. P_discharger = SPF * P_booster
        2. P_charger = P_discharger
        3. E_nom == full_load_hours * P_cap_discharger

    Args:
    -----
        n: PyPSA network object.
        model: The optimization model.
        SPF: The seasonal performance factor for the HT-ATES system.
        full_load_hours: Full load hours for the HT-ATES system.

    Returns:
    --------
        The updated optimization model with the new HT-ATES constraints.
    """
    chargers = get_assets_based_on_carrier_name(n, "Link", "HT-ATES-charger")
    dischargers = get_assets_based_on_carrier_name(n, "Link", "HT-ATES-discharger")
    booster_hps = get_assets_based_on_carrier_name(n, "Link", "HT-ATES-ASHP")
    stores = get_assets_based_on_carrier_name(n, "Store", "HT-ATES-store")

    if not (len(chargers) == len(dischargers) == len(booster_hps) == len(stores)):
        raise ValueError(
            f"Mismatch in HT-ATES asset counts: {len(chargers)} chargers,"
            f" {len(dischargers)} dischargers, {len(booster_hps)} booster heat pumps, {len(stores)} stores."
        )

    # 1. Add the seasonal performance factor constraint between discharger and booster heat pump
    spf_constraint = (
        model.variables["Link-p"].sel(name=dischargers) == SPF * model.variables["Link-p"].sel(name=booster_hps)
    )
    model.add_constraints(spf_constraint, name="ATES_discharger_booster_heat_pump_relation")

    # 2. Enforce charger and discharger capacities to be equal
    charger_caps = model.variables["Link-p_nom"].loc[chargers]
    discharger_caps = model.variables["Link-p_nom"].loc[dischargers]
    model.add_constraints(charger_caps == discharger_caps, name="ATES_charger_discharger_relation")

    # 3. Add power and energy capacity relation constraints
    e_nom_stores = model.variables["Store-e_nom"].loc[stores]
    p_nom_dischargers = model.variables["Link-p_nom"].loc[dischargers]
    model.add_constraints(e_nom_stores == full_load_hours * p_nom_dischargers, name="ATES_power_energy_relation")

    # # 4. Add constraint to ensure that if HT-ATES is built, it meets a minimum power capacity (LP -> MILP)
    # coords = {"name": dischargers}
    # dims = tuple(coords.keys())
    # model.add_variables(name="Link-build_ht_ates", coords=coords, dims=dims, binary=True)

    # # Upper bound: p_nom <= big_m * Link-build_ht_ates; Lower bound: p_nom >= min_power * Link-build_ht_ates
    # min_p_cap, big_m = 10, 1e3
    # p_nom_dischargers = model.variables["Link-p_nom"].loc[dischargers]
    # model.add_constraints(p_nom_dischargers <= big_m * model.variables["Link-build_ht_ates"], name="ATES_cap_ub")
    # model.add_constraints(p_nom_dischargers >= min_p_cap * model.variables["Link-build_ht_ates"], name="ATES_cap_lb")

    return model


def update_network_data(
    n: pypsa.Network,
    temperature: pd.DataFrame,
    heat_demand: pd.DataFrame,
    hydrogen_price: pd.DataFrame,
    electricity_price: pd.DataFrame,
    solar_availability: pd.DataFrame,
    static_prices: dict[str, float],
    snapshots: pd.DatetimeIndex,
) -> pypsa.Network:
    """Update network data before optimization.

    Updates the network data to assign prices, efficiency (COP), availability (p_max_pu),
    and demand timeseries to the network components.

    Args:
    -----
        n: PyPSA network object.
        temperature: DataFrame containing temperature timeseries.
        heat_demand: DataFrame containing heat demand timeseries.
        hydrogen_price: DataFrame containing hydrogen price timeseries.
        electricity_price: DataFrame containing electricity price timeseries.
        solar_availability: DataFrame containing solar availability timeseries for solar thermal generators.
        static_prices: DataFrame with non-time varying prices for greengas, waste material, residual heat, etc.
        snapshots: Set of timestamps to consider in the optimization.

    Returns:
    --------
        The PyPSA network object with updated data.
    """
    # Assign static prices
    year = snapshots[0].year
    n.stores_t.marginal_cost["greengas_source"] = static_prices.loc[year, "greengas"]
    n.stores_t.marginal_cost["waste_material_source"] = static_prices.loc[year, "waste_material"]

    ## Find generators with carrier name "Residual heat" and set their marginal costs
    residual_heat_generators = n.generators[n.generators.carrier == "Residual heat"].index
    n.generators.loc[residual_heat_generators, "marginal_cost"] = static_prices.loc[year, "residual_heat"]

    # Assign time-dependent prices
    n.generators_t.marginal_cost["electricity_supply"] = electricity_price.loc[snapshots, "electricity_price"].values
    n.stores_t.marginal_cost["hydrogen_source"] = hydrogen_price.loc[snapshots, "hydrogen_price"].values

    # Assign demand data to loads.
    for demand_profile in heat_demand.columns:
        n.loads_t.p_set[demand_profile] = heat_demand.loc[snapshots, demand_profile].values

    # Assign COP as efficiency of heat pumps
    ambient_temp = temperature.loc[snapshots, "ambient"].values
    heat_pump_cop = np.round(calculate_heatpump_cop(tech_carrier="ASHP", temp_source=ambient_temp), 3)

    # Get all ASHPs including the booster heat pumps of geothermal wells and HT-ATES systems
    heat_pump_assets = get_assets_based_on_carrier_name(n, component_type="Link", carrier_name="ASHP")
    heat_pump_assets += get_assets_based_on_carrier_name(n, component_type="Link", carrier_name="Geothermal ASHP")
    heat_pump_assets += get_assets_based_on_carrier_name(n, component_type="Link", carrier_name="HT-ATES-ASHP")
    for heatpump in heat_pump_assets:
        n.links_t.efficiency.loc[:, heatpump] = heat_pump_cop

    # Assign solar availability as p_max_pu of solar thermal generators
    for gen in get_assets_based_on_carrier_name(n, component_type="Generator", carrier_name="Solar thermal"):
        n.generators_t.p_max_pu[gen] = solar_availability.loc[snapshots, "availability"].values

    return n


def build_electricity_revenue(
    n: pypsa.Network, model: Model, chps_assets: list, electricity_price_vals: np.ndarray
) -> LinearExpression:
    """Build expression for electricity revenue from CHPs.

    NOTE: This function assumes that the CHPs generate electricity at bus2 of the
        pypsa network (efficiency2 is used to extract the electric efficiency.)

    Args:
    -----
        n: PyPSA network object.
        model: Linopy model object.
        chps_assets: List of CHP assets in the network.
        electricity_price_vals: Array containing electricity price timeseries.

    Returns:
    --------
        Linopy LinearExpression for electricity revenue from CHPs.
    """
    coords = {"snapshot": model.variables.coords["snapshot"]}
    electricity_price_xr = xr.DataArray(electricity_price_vals, coords=coords, dims=tuple(coords.keys()))

    electric_efficiency_chps = n.links.efficiency2.loc[chps_assets]
    gas_consumption_chps = model.variables["Link-p"].loc[:, chps_assets]

    electricity_generation_chps = gas_consumption_chps * electric_efficiency_chps
    electricity_revenue_from_chps = (electricity_price_xr * electricity_generation_chps).sum()

    return electricity_revenue_from_chps


def update_objective_function(model: Model, electricity_revenue_from_chps: LinearExpression) -> Model:
    """Update objective function to include electricity revenue from CHPs.

    Args:
    -----
        model: Linopy model object.
        electricity_revenue_from_chps: electricity revenue from CHPs.
    """
    new_objective = model.objective.expression - electricity_revenue_from_chps
    model.objective = new_objective
    return model


def add_volumetric_tou_tariffs(n: pypsa.Network, vol_tou_tariffs: pd.DataFrame) -> pypsa.Network:
    """Add volumetric TOU tariffs.

    This is done by simply adding the volumetric TOU tariff prices to the
    marginal price of electricity supply for each snapshot.

    Args:
    -----
        n: PyPSA network object.
        vol_tou_tariffs: DataFrame containing volumetric TOU tariff prices in €/MWh.

    Returns:
    --------
        The PyPSA network object with added volumetric TOU tariff links.
    """
    elec_cost_with_vol_tou_tariff = (
        n.generators_t.marginal_cost["electricity_supply"].values + vol_tou_tariffs["vol_tou_tariff"].values
    )

    n.generators_t.marginal_cost["electricity_supply"] = elec_cost_with_vol_tou_tariff

    return n
