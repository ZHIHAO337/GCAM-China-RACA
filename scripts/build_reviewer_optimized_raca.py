from __future__ import annotations

import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
DESIGN_PATH = PROJECT / "config/reviewer_optimized_design.json"
PROCESSED = PROJECT / "data/processed"
OPT_DATA = PROJECT / "data/optimized"
MODEL_XML = PROJECT / "model_xml/optimized"
QUERIES = PROJECT / "queries"
YEARS = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
HISTORICAL_YEARS = [1975, 1990, 2005, 2010, 2015, 2021]
SECTORS = ["base load generation", "intermediate generation", "subpeak generation", "peak generation"]
EJ_TO_TWH = 277.7777777778


def config_value(root: ET.Element, section: str, name: str) -> ET.Element:
    value = root.find(f"./{section}/Value[@name='{name}']")
    if value is None:
        raise KeyError(f"Configuration value not found: {section}/{name}")
    return value


def build_import_dependency(master: pd.DataFrame) -> pd.DataFrame:
    dependency = read_csv(PROCESSED / "province_import_dependency_bridge.csv")
    missing = set(master["province"]) - set(dependency["province"])
    if missing:
        raise ValueError(f"Missing province import-dependency rows: {sorted(missing)}")
    return dependency


def augment_risk(risk: pd.DataFrame, dependency: pd.DataFrame, design: dict) -> pd.DataFrame:
    result = risk.merge(
        dependency[["province", "net_import_dependency"]], on="province", how="left"
    )
    result["net_import_dependency"] = result["net_import_dependency"].fillna(0.0)
    reserve_margin = float(design["baseline_reserve_margin"])
    result["baseline_reserve_margin"] = reserve_margin
    result["event_transfer_availability"] = (1.0 - result["trade_loss"]).clip(0.0, 1.0)
    # Legacy field retained for old interfaces. Peak-load stress is applied once
    # when event peak is constructed, and trade loss acts only on transfer supply.
    result["equivalent_reserve_margin_with_trade"] = reserve_margin
    result["event_duration_hours"] = result["risk_level"].map(
        design["risk_event_duration_hours"]
    ).astype(float)
    result["risk_chain"] = (
        "hazard_return_period->delta_peak_to_event_peak+technology_derate_to_firm_capacity"
        "+trade_loss_to_transfer_supply;baseline_reserve_margin_applied_once"
    )
    return result


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", encoding="utf-8-sig")


def interpolate(start: float, end: float, year: int) -> float:
    return start + (end - start) * (year - 2025) / (2060 - 2025)


def build_storage_targets(master: pd.DataFrame, design: dict) -> pd.DataFrame:
    storage = design["storage"]
    consumption = master.groupby("grid.region")["electricity_consumption_2024_twh"].sum()
    rows = []
    for pathway, targets in storage["national_power_targets_gw"].items():
        end_mix = storage["technology_power_share_2060"][pathway]
        for year in YEARS:
            national_power = float(targets[str(year)])
            utilization = interpolate(
                float(storage["observed_2025_equivalent_utilization_hours"]),
                float(storage["equivalent_utilization_hours_2060"][pathway]),
                year,
            )
            mix = {
                str(duration): interpolate(
                    float(storage["technology_power_share_2025"][str(duration)]),
                    float(end_mix[str(duration)]),
                    year,
                )
                for duration in storage["durations_hours"]
            }
            mix_total = sum(mix.values())
            mix = {duration: value / mix_total for duration, value in mix.items()}
            for _, province in master.iterrows():
                grid = province["grid.region"]
                grid_share = float(storage["grid_region_power_shares_2024"][grid])
                province_share = float(province["electricity_consumption_2024_twh"] / consumption[grid])
                province_power = national_power * grid_share * province_share
                for duration in storage["durations_hours"]:
                    technology_share = mix[str(duration)]
                    power = province_power * technology_share
                    output_twh = power * utilization / 1000.0
                    rows.append(
                        {
                            "storage_pathway": pathway,
                            "province": province["province"],
                            "province.name": province["province.name"],
                            "province_zh": province["province_zh"],
                            "grid_region": grid,
                            "year": year,
                            "technology": f"battery_{duration}h",
                            "duration_hours": duration,
                            "technology_power_share": technology_share,
                            "target_power_gw": power,
                            "target_energy_gwh": power * duration,
                            "equivalent_utilization_hours": utilization,
                            "capacity_factor": utilization / 8760.0,
                            "fixed_output_twh": output_twh,
                            "fixed_output_ej": output_twh / EJ_TO_TWH,
                            "target_status": "observed_2025_anchor_and_policy_constrained_long_term_path",
                        }
                    )
    return pd.DataFrame(rows)


def build_storage_credits(risk: pd.DataFrame, design: dict) -> pd.DataFrame:
    availability = float(design["storage"]["availability"])
    rows = []
    for row in risk[[
        "province", "province.name", "province_zh", "climate_scenario",
        "risk_level", "year", "event_duration_hours",
    ]].itertuples(index=False):
        for duration in design["storage"]["durations_hours"]:
            event_duration = float(row.event_duration_hours)
            duration_credit = 1.0 if event_duration == 0 else min(1.0, duration / event_duration)
            rows.append(
                {
                    "climate_scenario": getattr(row, "climate_scenario", "ssp245"),
                    "province": row.province,
                    "province.name": getattr(row, "_1"),
                    "province_zh": row.province_zh,
                    "risk_level": row.risk_level,
                    "year": row.year,
                    "event_duration_hours": event_duration,
                    "technology": f"battery_{duration}h",
                    "duration_hours": duration,
                    "duration_credit": duration_credit,
                    "availability": availability,
                    "firm_capacity_credit": duration_credit * availability,
                }
            )
    return pd.DataFrame(rows)


def applied_derate(
    conditional: float,
    sector: str,
    category: str,
    risk_level: str,
    event_duration_hours: float,
    design: dict,
) -> float:
    if sector == "peak generation":
        return conditional
    if category == "hydro":
        hours = float(design["risk_application"]["hydro_drought_duration_days"][risk_level]) * 24.0
    else:
        hours = event_duration_hours
    return conditional * min(1.0, hours / 8760.0)


def add_derates(
    sector_xml: ET.Element,
    province: str,
    sector: str,
    risk_rows: pd.DataFrame,
    inventory: pd.DataFrame,
    design: dict,
) -> tuple[int, list[dict]]:
    count = 0
    records = []
    province_inv = inventory[
        inventory["province"].eq(province)
        & inventory["supplysector"].eq(sector)
        & (inventory["capacity_factor"].notna() | inventory["fixed_output_ej"].notna())
    ]
    category_column = {
        "pv": "derate_pv",
        "wind": "derate_wind",
        "hydro": "derate_hydro",
        "thermal": "derate_thermal",
    }
    for (subsector, technology), tech_rows in province_inv.groupby(["subsector", "technology"]):
        category = str(tech_rows["technology_category"].iloc[0])
        derate_column = category_column.get(category)
        if derate_column is None:
            continue
        sub = ET.SubElement(sector_xml, "nesting-subsector", {"name": subsector, "nocreate": "1"})
        tech = ET.SubElement(sub, "stub-technology", {"name": technology, "nocreate": "1"})
        for inv in tech_rows.itertuples(index=False):
            risk = risk_rows[risk_rows["year"].eq(inv.year)].iloc[0]
            conditional = float(risk[derate_column])
            derate = applied_derate(
                conditional,
                sector,
                category,
                str(risk["risk_level"]),
                float(risk["event_duration_hours"]),
                design,
            )
            period = ET.SubElement(tech, "period", {"year": str(inv.year)})
            original = np.nan
            adjusted = np.nan
            parameter = ""
            if not pd.isna(inv.capacity_factor):
                original = float(inv.capacity_factor)
                adjusted = max(0.005, original * (1.0 - derate))
                ET.SubElement(period, "capacity-factor").text = f"{adjusted:.9f}"
                parameter = "capacity-factor"
            elif category == "hydro" and not pd.isna(inv.fixed_output_ej):
                original = float(inv.fixed_output_ej)
                adjusted = max(0.0, original * (1.0 - derate))
                ET.SubElement(period, "fixedOutput").text = f"{adjusted:.12g}"
                parameter = "fixedOutput"
            else:
                continue
            records.append(
                {
                    "province": province,
                    "sector": sector,
                    "subsector": subsector,
                    "technology": technology,
                    "technology_category": category,
                    "risk_level": risk["risk_level"],
                    "year": inv.year,
                    "conditional_event_derate": conditional,
                    "gcam_applied_derate": derate,
                    "event_duration_hours": risk["event_duration_hours"],
                    "parameter": parameter,
                    "original_value": original,
                    "adjusted_value": adjusted,
                }
            )
            count += 1
    return count, records


def add_global_storage_technologies(
    world: ET.Element,
    pathway: str,
    storage_costs: pd.DataFrame,
    targets: pd.DataFrame,
    design: dict,
) -> None:
    database = ET.SubElement(world, "global-technology-database")
    location = ET.SubElement(
        database,
        "location-info",
        {"sector-name": "peak generation", "subsector-name": "grid_storage"},
    )
    for duration in design["storage"]["durations_hours"]:
        technology = f"battery_{duration}h"
        tech = ET.SubElement(location, "technology", {"name": technology})
        for year in HISTORICAL_YEARS + YEARS:
            parameter_year = 2025 if year < 2025 else year
            cost_pathway = "S1_ORDERLY" if parameter_year == 2025 else pathway
            cost = storage_costs[
                storage_costs["storage_pathway"].eq(cost_pathway)
                & storage_costs["technology"].eq(technology)
                & storage_costs["year"].eq(parameter_year)
            ].iloc[0]
            target = targets[
                targets["storage_pathway"].eq(pathway)
                & targets["technology"].eq(technology)
                & targets["year"].eq(parameter_year)
            ]
            capacity_factor = (
                float(target["capacity_factor"].iloc[0])
                if not target.empty
                else float(design["storage"]["observed_2025_equivalent_utilization_hours"]) / 8760.0
            )
            share_weight = 0.0 if year < 2025 else float(target["technology_power_share"].iloc[0])
            period = ET.SubElement(tech, "period", {"year": str(year)})
            ET.SubElement(period, "share-weight").text = f"{share_weight:.9f}"
            ET.SubElement(period, "CO2", {"name": "CO2"})
            capital = ET.SubElement(period, "input-capital", {"name": "capital"})
            ET.SubElement(capital, "capital-overnight").text = f"{cost['capex_1975usd_per_kw']:.9f}"
            ET.SubElement(capital, "fixed-charge-rate").text = f"{cost['fixed_charge_rate']:.6f}"
            fixed_om = ET.SubElement(period, "input-OM-fixed", {"name": "OM-fixed"})
            ET.SubElement(fixed_om, "OM-fixed").text = f"{cost['fixed_om_1975usd_per_kw_year']:.9f}"
            variable_om = ET.SubElement(period, "input-OM-var", {"name": "OM-var"})
            ET.SubElement(variable_om, "OM-var").text = "0"
            ET.SubElement(period, "capacity-factor").text = f"{capacity_factor:.9f}"
            if year >= 2025:
                ET.SubElement(period, "lifetime").text = str(int(cost["lifetime_years"]))


def add_storage(
    peak: ET.Element,
    province: pd.Series,
    pathway: str,
    targets: pd.DataFrame,
    design: dict,
) -> None:
    sub = ET.SubElement(peak, "nesting-subsector", {"name": "grid_storage", "nocreate": "1"})
    delete = ET.SubElement(
        sub,
        "interpolation-rule",
        {"apply-to": "share-weight", "delete": "1", "from-year": "2021", "to-year": "2100"},
    )
    ET.SubElement(delete, "interpolation-function", {"name": "linear"})
    for year in YEARS:
        ET.SubElement(sub, "share-weight", {"year": str(year)}).text = "1"
    original = ET.SubElement(sub, "stub-technology", {"name": "battery", "nocreate": "1"})
    for year in YEARS:
        period = ET.SubElement(original, "period", {"year": str(year)})
        ET.SubElement(period, "share-weight").text = "0"
    for duration in design["storage"]["durations_hours"]:
        technology = f"battery_{duration}h"
        tech = ET.SubElement(sub, "stub-technology", {"name": technology})
        province_targets = targets[
            targets["storage_pathway"].eq(pathway)
            & targets["province"].eq(province["province"])
            & targets["technology"].eq(technology)
        ]
        for year in HISTORICAL_YEARS + YEARS:
            period = ET.SubElement(tech, "period", {"year": str(year)})
            energy = ET.SubElement(period, "minicam-energy-input", {"name": "off peak electricity"})
            ET.SubElement(energy, "efficiency").text = f"{design['storage']['round_trip_efficiency']:.6f}"
            ET.SubElement(energy, "market-name").text = str(province["grid.region"])
            if year >= 2025:
                target = province_targets[province_targets["year"].eq(year)].iloc[0]
                ET.SubElement(period, "fixedOutput").text = f"{target['fixed_output_ej']:.12g}"


def build_components(
    master: pd.DataFrame,
    scenarios: pd.DataFrame,
    risk: pd.DataFrame,
    inventory: pd.DataFrame,
    storage_costs: pd.DataFrame,
    targets: pd.DataFrame,
    design: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    MODEL_XML.mkdir(parents=True, exist_ok=True)
    components = []
    derate_records = []
    for scenario in scenarios.itertuples(index=False):
        root = ET.Element("scenario")
        world = ET.SubElement(root, "world")
        add_global_storage_technologies(world, scenario.storage_pathway, storage_costs, targets, design)
        count = 0
        for _, province in master.iterrows():
            code = province["province"]
            region = ET.SubElement(world, "region", {"name": code})
            risk_rows = risk[
                risk["province"].eq(code) & risk["risk_level"].eq(scenario.risk_level)
            ]
            sectors = {}
            for sector in SECTORS:
                sectors[sector] = ET.SubElement(
                    region, "pass-through-sector", {"name": sector, "nocreate": "1"}
                )
                added, records = add_derates(
                    sectors[sector], code, sector, risk_rows, inventory, design
                )
                count += added
                for record in records:
                    record["scenario_id"] = scenario.scenario_id
                derate_records.extend(records)
            add_storage(sectors["peak generation"], province, scenario.storage_pathway, targets, design)
        ET.indent(root, space="    ")
        path = MODEL_XML / f"optimized_{scenario.scenario_id.lower()}.xml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        components.append(
            {
                "scenario_id": scenario.scenario_id,
                "component_xml": str(path),
                "province_count": len(master),
                "derate_values": count,
            }
        )
    return pd.DataFrame(components), pd.DataFrame(derate_records)


def write_batch(path: Path, components: pd.DataFrame, name: str) -> None:
    root = ET.Element("BatchRunner")
    component_set = ET.SubElement(root, "ComponentSet", {"name": name})
    for row in components.itertuples(index=False):
        file_set = ET.SubElement(component_set, "FileSet", {"name": f"_{row.scenario_id}"})
        value = ET.SubElement(file_set, "Value", {"name": "raca_optimized_component"})
        value.text = "../" + Path(row.component_xml).resolve().relative_to(REPO.resolve()).as_posix()
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_configuration(
    path: Path, batch: Path, database: str, stop_year: int, scenario_prefix: str = "RACA_OPT"
) -> None:
    tree = ET.parse(REPO / "exe/configuration_china.xml")
    root = tree.getroot()
    config_value(root, "Files", "BatchFileName").text = (
        "../" + batch.resolve().relative_to(REPO.resolve()).as_posix()
    )
    config_value(root, "Files", "xmldb-location").text = database
    config_value(root, "Files", "restart").set("write-output", "0")
    config_value(root, "Files", "xmlDebugFileName").set("write-output", "0")
    config_value(root, "Files", "climatFileName").set("write-output", "0")
    config_value(root, "Strings", "scenarioName").text = scenario_prefix
    config_value(root, "Bools", "BatchMode").text = "1"
    config_value(root, "Bools", "QuitFirstFailure").text = "1"
    config_value(root, "Ints", "stop-year").text = str(stop_year)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_query_batch(
    path: Path,
    scenarios: list[str],
    database: str,
    outfile: str,
    scenario_prefix: str = "RACA_OPT",
) -> None:
    root = ET.Element("ModelInterfaceBatch")
    klass = ET.SubElement(root, "class", {"name": "ModelInterface.ModelGUI2.DbViewer"})
    command = ET.SubElement(klass, "command", {"name": "XMLDB Batch File"})
    for scenario in scenarios:
        ET.SubElement(command, "scenario", {"name": f"{scenario_prefix}_{scenario}"})
    ET.SubElement(command, "queryFile").text = "gcam_raca/queries/raca_current_queries.xml"
    ET.SubElement(command, "outFile").text = outfile
    ET.SubElement(command, "xmldbLocation").text = database
    ET.SubElement(command, "batchQueryResultsInDifferentSheets").text = "false"
    ET.SubElement(command, "batchQueryIncludeCharts").text = "false"
    ET.SubElement(command, "batchQuerySplitRunsInDifferentSheets").text = "false"
    ET.SubElement(command, "batchQueryReplaceResults").text = "true"
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    global MODEL_XML, OPT_DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--climate-scenario", choices=["ssp245", "ssp585"], default="ssp245")
    parser.add_argument("--scenario-prefix", default="RACA_OPT")
    parser.add_argument("--model-xml", type=Path, default=MODEL_XML)
    parser.add_argument("--data-output", type=Path, default=OPT_DATA)
    parser.add_argument(
        "--database", default="../gcam_raca/outputs/optimized/model_database/raca_optimized_12"
    )
    parser.add_argument("--query-output", default="gcam_raca/outputs/optimized/raca_12_query_results.csv")
    parser.add_argument("--query-batch", type=Path, default=QUERIES / "query_optimized_12_batch.xml")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    MODEL_XML = args.model_xml.resolve()
    OPT_DATA = args.data_output.resolve()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    master = read_csv(PROCESSED / "province_master.csv")
    scenarios = pd.DataFrame([
        {
            "scenario_id": f"{risk_level}_{storage_pathway}",
            "risk_level": risk_level,
            "storage_pathway": storage_pathway,
        }
        for risk_level in ["R0", "R10", "R20", "R50"]
        for storage_pathway in ["S0_BASE", "S1_ORDERLY", "S2_ACCELERATED"]
    ])
    risk = read_csv(PROJECT / f"data/formal/risk_parameters_{args.climate_scenario}.csv")
    inventory = read_csv(PROCESSED / "gcam_peak_technology_inventory.csv")
    storage_costs = read_csv(PROCESSED / "storage_technoeconomic_atb2024.csv")
    dependency = build_import_dependency(master)
    risk = augment_risk(risk, dependency, {
        "baseline_reserve_margin": design["baseline_reserve_margin"],
        "risk_event_duration_hours": design["risk_event_duration_hours"],
    })
    risk["climate_scenario"] = args.climate_scenario
    targets = build_storage_targets(master, design)
    credits = build_storage_credits(risk, design)
    components, derates = build_components(
        master, scenarios, risk, inventory, storage_costs, targets, design
    )
    components["climate_scenario"] = args.climate_scenario
    derates["climate_scenario"] = args.climate_scenario
    OPT_DATA.mkdir(parents=True, exist_ok=True)
    targets.to_csv(OPT_DATA / "storage_deployment_targets.csv", index=False, encoding="utf-8-sig")
    credits.to_csv(OPT_DATA / "storage_firm_capacity_credit.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(OPT_DATA / "risk_parameters_optimized_interface.csv", index=False, encoding="utf-8-sig")
    derates.to_csv(OPT_DATA / "gcam_applied_derates.csv", index=False, encoding="utf-8-sig")
    components.to_csv(OPT_DATA / "generated_optimized_components.csv", index=False, encoding="utf-8-sig")

    batch = MODEL_XML / f"batch_{args.climate_scenario}_12_scenarios.xml"
    write_batch(batch, components, "GCAM-China-RACA-reviewer-optimized")
    write_configuration(
        MODEL_XML / "configuration_optimized_12_scenarios_2060.xml",
        batch,
        args.database,
        2060,
        args.scenario_prefix,
    )
    if not args.skip_smoke:
        smoke = components[components["scenario_id"].isin(["R0_S1_ORDERLY", "R50_S1_ORDERLY"])]
        smoke_batch = MODEL_XML / f"batch_{args.climate_scenario}_smoke.xml"
        write_batch(smoke_batch, smoke, "GCAM-China-RACA-reviewer-optimized-smoke")
        write_configuration(
            MODEL_XML / f"configuration_{args.climate_scenario}_smoke_2030.xml",
            smoke_batch,
            "../gcam_raca/outputs/v3/smoke_database",
            2030,
            args.scenario_prefix,
        )
    write_query_batch(
        args.query_batch,
        scenarios["scenario_id"].tolist(),
        args.database.removeprefix("../"),
        args.query_output,
        args.scenario_prefix,
    )
    print(
        f"Built {len(components)} optimized components, {len(targets)} storage targets, "
        f"{len(credits)} capacity-credit rows and {len(derates)} applied derates."
    )


if __name__ == "__main__":
    main()
