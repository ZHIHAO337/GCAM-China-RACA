from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
DEFAULT_INPUT = PROJECT / "outputs/formal/raca_12_query_results.csv"
OUT = PROJECT / "outputs/formal/processed"
EJ_TO_TWH = 277.7777777778
POWER_SECTORS = {"base load generation", "intermediate generation", "subpeak generation", "peak generation"}
MODEL_YEARS = {2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060}


def read_query_sections(path: Path) -> list[tuple[str, list[str], list[list[str]]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = list(csv.reader(stream))
    sections = []
    index = 0
    while index < len(rows):
        nonempty = [value for value in rows[index] if value.strip()]
        if len(nonempty) != 1 or index + 1 >= len(rows):
            index += 1
            continue
        title = nonempty[0].strip()
        header = [value.strip() for value in rows[index + 1]]
        while header and not header[-1]:
            header.pop()
        data_rows = []
        index += 2
        while index < len(rows):
            candidate = rows[index]
            candidate_nonempty = [value for value in candidate if value.strip()]
            if len(candidate_nonempty) == 1 and index + 1 < len(rows):
                next_first = rows[index + 1][0].strip() if rows[index + 1] else ""
                if next_first == "scenario":
                    break
            values = candidate[:len(header)] + [""] * max(0, len(header) - len(candidate))
            if any(value.strip() for value in values):
                data_rows.append(values)
            index += 1
        sections.append((title, header, data_rows))
    return sections


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_long(section, value_name: str) -> pd.DataFrame:
    _, header, rows = section
    years = [column for column in header if column.isdigit()]
    records = []
    for values in rows:
        row = dict(zip(header, values))
        scenario = row.get("scenario", "").split(",date=", 1)[0].strip().strip('"')
        if not (scenario.startswith("RACA_OPT_") or scenario.startswith("RACA_V3_")):
            continue
        common = {key: row.get(key, "") for key in [
            "region", "sector", "subsector", "technology", "input", "fuel", "account"
        ] if key in header}
        for year in years:
            records.append({"scenario": scenario, "year": int(year), value_name: number(row.get(year, "")), **common})
    return pd.DataFrame(records)


def parse_scenario(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = frame["scenario"].str.extract(
        r"RACA_OPT_(R0|R10|R20|R50)_(S0_BASE|S1_ORDERLY|S2_ACCELERATED)"
    )
    current = frame["scenario"].str.extract(
        r"RACA_V3_(SSP245|SSP585)_(R0|R10|R20|R50)_(S0_BASE|S1_ORDERLY|S2_ACCELERATED)"
    )
    frame = frame.copy()
    frame["climate_scenario"] = current[0].str.lower().fillna("ssp245")
    frame["risk_level"] = current[1].fillna(legacy[0])
    frame["storage_pathway"] = current[2].fillna(legacy[1])
    return frame


def classify_technology(value: str) -> str:
    text = str(value).lower()
    if text.startswith("battery_") or text == "battery":
        return "storage"
    if "wind" in text:
        return "wind"
    if any(token in text for token in ["pv", "solar", "csp"]):
        return "solar"
    if "hydro" in text:
        return "hydro"
    if any(token in text for token in ["coal", "gas", "liquid", "nuclear", "biomass"]):
        return "thermal"
    return "other"


def build_metrics(
    sections,
    storage_targets_path: Path | None = None,
    credits_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(sections) != 7:
        raise RuntimeError(f"Expected 7 query sections, found {len(sections)}")
    generation = parse_scenario(to_long(sections[0], "output_ej"))
    inputs = parse_scenario(to_long(sections[1], "input_ej"))
    prices = parse_scenario(to_long(sections[2], "price_1975usd_per_gj"))
    emissions = parse_scenario(to_long(sections[5], "co2_mtc"))
    water = parse_scenario(to_long(sections[6], "water_input"))

    generation = generation[generation["year"].isin(MODEL_YEARS)]
    inputs = inputs[inputs["year"].isin(MODEL_YEARS)]
    prices = prices[prices["year"].isin(MODEL_YEARS)]
    emissions = emissions[emissions["year"].isin(MODEL_YEARS)]
    water = water[water["year"].isin(MODEL_YEARS)]

    generation = generation[generation["sector"].isin(POWER_SECTORS)].copy()
    generation["technology_category"] = generation["technology"].map(classify_technology)
    generation["output_twh"] = generation["output_ej"] * EJ_TO_TWH
    keys = [
        "scenario", "climate_scenario", "risk_level", "storage_pathway", "region", "year"
    ]
    totals = generation.groupby(keys, as_index=False)["output_twh"].sum().rename(columns={"output_twh": "generation_twh"})
    categories = generation.pivot_table(
        index=keys, columns="technology_category", values="output_twh", aggfunc="sum", fill_value=0
    ).reset_index()
    categories.columns.name = None
    metrics = totals.merge(categories, on=keys, how="left")
    for column in ["storage", "wind", "solar", "hydro", "thermal", "other"]:
        if column not in metrics:
            metrics[column] = 0.0
    metrics = metrics.rename(columns={
        "storage": "storage_discharge_twh", "wind": "wind_generation_twh",
        "solar": "solar_generation_twh", "hydro": "hydro_generation_twh",
        "thermal": "thermal_generation_twh",
    })
    metrics["vre_generation_twh"] = metrics["wind_generation_twh"] + metrics["solar_generation_twh"]

    storage_input = inputs[inputs["technology"].str.startswith("battery_", na=False)].copy()
    charging = storage_input.groupby(keys, as_index=False)["input_ej"].sum()
    charging["storage_charge_twh"] = charging["input_ej"] * EJ_TO_TWH
    metrics = metrics.merge(charging[keys + ["storage_charge_twh"]], on=keys, how="left")

    price_subset = prices[prices.get("fuel", "").isin(["electricity", "peak generation"])].copy()
    if price_subset.empty:
        price_subset = prices
    price = price_subset.groupby(keys, as_index=False)["price_1975usd_per_gj"].mean()
    metrics = metrics.merge(price, on=keys, how="left")

    co2 = emissions[emissions["sector"].isin(POWER_SECTORS)].groupby(keys, as_index=False)["co2_mtc"].sum()
    water_total = water[water["sector"].isin(POWER_SECTORS)].groupby(keys, as_index=False)["water_input"].sum()
    metrics = metrics.merge(co2, on=keys, how="left").merge(water_total, on=keys, how="left")

    storage_tech = generation[generation["technology_category"] == "storage"].copy()
    storage_tech["duration_hours"] = pd.to_numeric(
        storage_tech["technology"].str.extract(r"battery_(\d+)h")[0], errors="coerce"
    )
    credits = pd.read_csv(
        credits_path or PROJECT / "data/processed/storage_firm_capacity_credit.csv",
        encoding="utf-8-sig",
    )
    storage_tech = storage_tech.merge(
        credits[[
            "province", "climate_scenario", "risk_level", "year", "technology",
            "firm_capacity_credit",
        ]],
        left_on=["region", "climate_scenario", "risk_level", "year", "technology"],
        right_on=["province", "climate_scenario", "risk_level", "year", "technology"], how="left"
    )
    if storage_targets_path is not None:
        targets = pd.read_csv(storage_targets_path, encoding="utf-8-sig")
        storage_tech = storage_tech.merge(
            targets[[
                "storage_pathway", "province", "year", "technology", "target_power_gw",
                "target_energy_gwh", "capacity_factor", "equivalent_utilization_hours",
            ]],
            left_on=["storage_pathway", "region", "year", "technology"],
            right_on=["storage_pathway", "province", "year", "technology"],
            how="left",
            suffixes=("", "_target"),
        )
        storage_tech["effective_capacity_factor"] = storage_tech["capacity_factor"]
        storage_tech["storage_power_gw"] = storage_tech["target_power_gw"]
        storage_tech["storage_energy_gwh"] = storage_tech["target_energy_gwh"]
    else:
        storage_tech["effective_capacity_factor"] = (
            storage_tech["duration_hours"] / 24.0 * storage_tech["firm_capacity_credit"]
        )
        storage_tech["storage_power_gw"] = (
            storage_tech["output_twh"]
            / (8.76 * storage_tech["effective_capacity_factor"].replace(0, np.nan))
        )
        storage_tech["storage_energy_gwh"] = (
            storage_tech["storage_power_gw"] * storage_tech["duration_hours"]
        )
    storage_capacity = storage_tech.groupby(keys, as_index=False)[["storage_power_gw", "storage_energy_gwh"]].sum()
    metrics = metrics.merge(storage_capacity, on=keys, how="left")
    metrics["primary_generation_twh"] = metrics["generation_twh"] - metrics["storage_discharge_twh"]
    metrics["net_supply_twh"] = metrics["generation_twh"] - metrics["storage_charge_twh"].fillna(0)
    metrics["vre_share_gross_generation"] = (
        metrics["vre_generation_twh"] / metrics["generation_twh"].replace(0, np.nan)
    )
    metrics["vre_share"] = (
        metrics["vre_generation_twh"] / metrics["primary_generation_twh"].replace(0, np.nan)
    )
    metrics = metrics[metrics["generation_twh"].gt(1e-9)].copy()
    solved_keys = metrics[keys].drop_duplicates()
    storage_tech = storage_tech.merge(solved_keys, on=keys, how="inner")
    numeric = metrics.select_dtypes(include="number").columns
    metrics[numeric] = metrics[numeric].fillna(0)
    return metrics.sort_values(keys), storage_tech


def decompose(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "generation_twh", "primary_generation_twh", "net_supply_twh", "storage_discharge_twh",
        "storage_charge_twh", "storage_power_gw", "storage_energy_gwh", "vre_generation_twh",
        "vre_share", "price_1975usd_per_gj", "co2_mtc", "water_input",
    ]
    scenario_lookup = {
        "R0S0": ("R0", "S0_BASE"), "R0S2": ("R0", "S2_ACCELERATED"),
        "R50S0": ("R50", "S0_BASE"), "R50S2": ("R50", "S2_ACCELERATED"),
    }
    rows = []
    for (climate_scenario, region, year), group in metrics.groupby(
        ["climate_scenario", "region", "year"]
    ):
        indexed = group.set_index(["risk_level", "storage_pathway"])
        if not all(key in indexed.index for key in scenario_lookup.values()):
            continue
        for metric in value_columns:
            values = {name: float(indexed.loc[key, metric]) for name, key in scenario_lookup.items()}
            rows.append({
                "climate_scenario": climate_scenario,
                "region": region, "year": year, "metric": metric,
                "risk_effect": values["R50S0"] - values["R0S0"],
                "storage_effect": values["R0S2"] - values["R0S0"],
                "buffer_effect": values["R50S2"] - values["R50S0"],
                "interaction_effect": (values["R50S2"] - values["R50S0"]) - (values["R0S2"] - values["R0S0"]),
            })
    return pd.DataFrame(rows)


def classify_provinces(metrics: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame(columns=["region", "province_type", "classification_status"])
    hazard_long = pd.read_csv(
        PROJECT / "data/formal/cmip6_node_hazard_ensemble.csv", encoding="utf-8-sig"
    )
    dependency = pd.read_csv(PROJECT / "data/processed/province_import_dependency_bridge.csv", encoding="utf-8-sig")
    tables = []
    for climate_scenario in sorted(metrics["climate_scenario"].dropna().unique()):
        hazard_subset = hazard_long[
            hazard_long["scenario"].eq(climate_scenario)
            & hazard_long["node_year"].eq(2060)
        ]
        hazard = hazard_subset.pivot_table(
            index="province", columns="hazard", values="median", aggfunc="first"
        ).reset_index()
        hazard_columns = [
            column for column in ["heat", "cold", "flood", "drought", "wind", "solar"]
            if column in hazard.columns
        ]
        hazard["compound_hazard_score"] = hazard[hazard_columns].rank(pct=True).mean(axis=1)
        hazard["drought_score"] = hazard["drought"].rank(pct=True)
        target = metrics[
            metrics["climate_scenario"].eq(climate_scenario)
            & metrics["year"].eq(2060)
            & metrics["risk_level"].eq("R50")
            & metrics["storage_pathway"].eq("S2_ACCELERATED")
        ]
        table = target[[
            "climate_scenario", "region", "storage_power_gw", "vre_share",
            "hydro_generation_twh",
        ]].merge(
            hazard[["province", "compound_hazard_score", "drought_score"]],
            left_on="region", right_on="province", how="left"
        ).merge(dependency[["province", "net_import_dependency"]], on="province", how="left")
        medians = table[["storage_power_gw", "vre_share", "hydro_generation_twh"]].median()

        def choose(row):
            if (
                row["compound_hazard_score"] >= 0.67
                and row["storage_power_gw"] >= medians["storage_power_gw"]
            ):
                return "高风险—高储能需求型"
            if row["compound_hazard_score"] >= 0.67 and row["net_import_dependency"] >= 0.20:
                return "高风险—跨省支援依赖型"
            if row["vre_share"] >= max(0.35, medians["vre_share"]):
                return "高新能源—长时储能需求型"
            if (
                row["drought_score"] >= 0.67
                and row["hydro_generation_twh"] >= medians["hydro_generation_twh"]
            ):
                return "水电干旱敏感型"
            return "一般转型型"

        table["province_type"] = table.apply(choose, axis=1)
        table["classification_status"] = (
            "formal_dual_ssp_cmip6_ensemble_and_complete_scenario_2060"
        )
        tables.append(table.drop(columns="province"))
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--storage-targets", type=Path)
    parser.add_argument("--credits", type=Path)
    args = parser.parse_args()
    source = args.input
    if not source.exists():
        raise FileNotFoundError(source)
    sections = read_query_sections(source)
    metrics, storage_tech = build_metrics(
        sections,
        storage_targets_path=args.storage_targets,
        credits_path=args.credits,
    )
    effects = decompose(metrics)
    province_types = classify_provinces(metrics, effects)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "province_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    storage_tech.to_csv(args.output_dir / "storage_technology_detail.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(args.output_dir / "content3_effect_decomposition.csv", index=False, encoding="utf-8-sig")
    province_types.to_csv(args.output_dir / "province_typology.csv", index=False, encoding="utf-8-sig")
    print(f"Processed {len(metrics)} province-scenario-year rows; decomposition rows: {len(effects)}")


if __name__ == "__main__":
    main()
