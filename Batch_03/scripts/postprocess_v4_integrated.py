from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import postprocess_raca as base  # noqa: E402


MODEL_YEARS = sorted(base.MODEL_YEARS)
SCENARIO_RE = re.compile(
    r"RACA_V4_(SSP245|SSP585)(?:_SMOKE)?_"
    r"(CN_NDC7_NZ2060|CN_NDC10_NZ2060|IPCC15_ACCELERATED_2050)_"
    r"(R0|R50)_(S0_BASE|S1_ORDERLY|S2_ACCELERATED)"
)


def to_long(section, value_name: str) -> pd.DataFrame:
    _, header, rows = section
    years = [column for column in header if column.isdigit()]
    records = []
    for values in rows:
        row = dict(zip(header, values))
        scenario = row.get("scenario", "").split(",date=", 1)[0].strip().strip('"')
        match = SCENARIO_RE.search(scenario)
        if not match:
            continue
        common = {
            key: row.get(key, "")
            for key in ("region", "sector", "subsector", "technology", "input", "fuel", "account")
            if key in header
        }
        dimensions = {
            "scenario": scenario,
            "climate_scenario": match.group(1).lower(),
            "ambition_pathway": match.group(2),
            "risk_level": match.group(3),
            "storage_pathway": match.group(4),
        }
        for year in years:
            if int(year) in base.MODEL_YEARS:
                records.append(
                    {
                        **dimensions,
                        **common,
                        "year": int(year),
                        value_name: base.number(row.get(year, "")),
                    }
                )
    return pd.DataFrame(records)


def build_metrics(sections, climate: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(sections) != 8:
        raise RuntimeError(f"Expected 8 query sections, found {len(sections)}")
    generation = to_long(sections[0], "output_ej")
    inputs = to_long(sections[1], "input_ej")
    prices = to_long(sections[2], "price_1975usd_per_gj")
    emissions = to_long(sections[5], "co2_mtc")
    water = to_long(sections[6], "water_input")
    all_co2 = to_long(sections[7], "co2_mtc")
    if generation.empty:
        raise RuntimeError("No V4 scenarios were parsed from query output")

    keys = [
        "scenario", "climate_scenario", "ambition_pathway", "risk_level",
        "storage_pathway", "region", "year",
    ]
    generation = generation[generation["sector"].isin(base.POWER_SECTORS)].copy()
    generation["technology_category"] = generation["technology"].map(base.classify_technology)
    generation["output_twh"] = generation["output_ej"] * base.EJ_TO_TWH
    totals = generation.groupby(keys, as_index=False)["output_twh"].sum().rename(
        columns={"output_twh": "generation_twh"}
    )
    categories = generation.pivot_table(
        index=keys, columns="technology_category", values="output_twh", aggfunc="sum", fill_value=0
    ).reset_index()
    categories.columns.name = None
    metrics = totals.merge(categories, on=keys, how="left")
    for column in ("storage", "wind", "solar", "hydro", "thermal", "other"):
        if column not in metrics:
            metrics[column] = 0.0
    metrics = metrics.rename(
        columns={
            "storage": "storage_discharge_twh",
            "wind": "wind_generation_twh",
            "solar": "solar_generation_twh",
            "hydro": "hydro_generation_twh",
            "thermal": "thermal_generation_twh",
        }
    )
    metrics["vre_generation_twh"] = metrics["wind_generation_twh"] + metrics["solar_generation_twh"]
    nonfossil_mask = generation["technology"].str.lower().str.contains(
        "wind|solar|pv|csp|hydro|nuc|biomass", regex=True, na=False
    )
    nonfossil = generation[nonfossil_mask].groupby(keys, as_index=False)["output_twh"].sum().rename(
        columns={"output_twh": "nonfossil_generation_twh"}
    )
    metrics = metrics.merge(nonfossil, on=keys, how="left")

    charging = inputs[inputs["technology"].str.startswith("battery_", na=False)].groupby(
        keys, as_index=False
    )["input_ej"].sum()
    charging["storage_charge_twh"] = charging["input_ej"] * base.EJ_TO_TWH
    metrics = metrics.merge(charging[keys + ["storage_charge_twh"]], on=keys, how="left")

    price_subset = prices[prices.get("fuel", "").isin(["electricity", "peak generation"])].copy()
    if price_subset.empty:
        price_subset = prices
    price = price_subset.groupby(keys, as_index=False)["price_1975usd_per_gj"].mean()
    power_co2 = emissions[emissions["sector"].isin(base.POWER_SECTORS)].groupby(
        keys, as_index=False
    )["co2_mtc"].sum()
    power_water = water[water["sector"].isin(base.POWER_SECTORS)].groupby(
        keys, as_index=False
    )["water_input"].sum()
    metrics = metrics.merge(price, on=keys, how="left").merge(
        power_co2, on=keys, how="left"
    ).merge(power_water, on=keys, how="left")

    storage_tech = generation[generation["technology_category"].eq("storage")].copy()
    storage_tech["duration_hours"] = pd.to_numeric(
        storage_tech["technology"].str.extract(r"battery_(\d+)h")[0], errors="coerce"
    )
    targets = pd.read_csv(
        ROOT / "data" / "v3" / climate / "storage_deployment_targets.csv", encoding="utf-8-sig"
    )
    credits = pd.read_csv(
        ROOT / "data" / "v3" / climate / "storage_firm_capacity_credit.csv", encoding="utf-8-sig"
    )
    storage_tech = storage_tech.merge(
        targets[
            ["storage_pathway", "province", "year", "technology", "target_power_gw", "target_energy_gwh"]
        ],
        left_on=["storage_pathway", "region", "year", "technology"],
        right_on=["storage_pathway", "province", "year", "technology"],
        how="left",
    ).merge(
        credits[["province", "climate_scenario", "risk_level", "year", "technology", "firm_capacity_credit"]],
        left_on=["region", "climate_scenario", "risk_level", "year", "technology"],
        right_on=["province", "climate_scenario", "risk_level", "year", "technology"],
        how="left",
        suffixes=("", "_credit"),
    )
    storage_tech["storage_power_gw"] = storage_tech["target_power_gw"]
    storage_tech["storage_energy_gwh"] = storage_tech["target_energy_gwh"]
    storage_capacity = storage_tech.groupby(keys, as_index=False)[
        ["storage_power_gw", "storage_energy_gwh"]
    ].sum()
    metrics = metrics.merge(storage_capacity, on=keys, how="left")
    metrics["primary_generation_twh"] = metrics["generation_twh"] - metrics["storage_discharge_twh"]
    metrics["net_supply_twh"] = metrics["generation_twh"] - metrics["storage_charge_twh"].fillna(0.0)
    metrics["vre_share_gross_generation"] = metrics["vre_generation_twh"] / metrics[
        "generation_twh"
    ].replace(0, np.nan)
    metrics["vre_share"] = metrics["vre_generation_twh"] / metrics[
        "primary_generation_twh"
    ].replace(0, np.nan)
    metrics["nonfossil_electricity_share"] = metrics["nonfossil_generation_twh"] / metrics[
        "primary_generation_twh"
    ].replace(0, np.nan)
    numeric = metrics.select_dtypes(include="number").columns
    metrics[numeric] = metrics[numeric].fillna(0.0)
    metrics = metrics[metrics["generation_twh"].gt(1e-9)].copy()

    national_co2 = all_co2.groupby(
        ["scenario", "climate_scenario", "ambition_pathway", "risk_level", "storage_pathway", "year"],
        as_index=False,
    )["co2_mtc"].sum()
    caps = pd.read_csv(ROOT / "data" / "v4" / "ambition_policy_constraints.csv", encoding="utf-8-sig")
    national_co2 = national_co2.merge(caps, on=["ambition_pathway", "year"], how="left")
    national_co2["cap_slack_mtc"] = national_co2["co2_constraint_mtc"] - national_co2["co2_mtc"]
    national_co2["cap_binding"] = national_co2["cap_slack_mtc"].abs().le(
        national_co2["co2_constraint_mtc"].clip(lower=1.0) * 0.002
    )
    return metrics.sort_values(keys), storage_tech, national_co2


def effects(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_columns = [
        "net_supply_twh", "vre_generation_twh", "vre_share", "storage_power_gw",
        "price_1975usd_per_gj", "co2_mtc", "water_input",
    ]
    group_keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    for key, group in metrics.groupby(group_keys):
        indexed = group.set_index(["risk_level", "storage_pathway"])
        needed = [("R0", "S1_ORDERLY"), ("R50", "S0_BASE"), ("R50", "S1_ORDERLY"), ("R50", "S2_ACCELERATED")]
        if not all(item in indexed.index for item in needed):
            continue
        for metric in value_columns:
            r0s1 = float(indexed.loc[("R0", "S1_ORDERLY"), metric])
            r50s0 = float(indexed.loc[("R50", "S0_BASE"), metric])
            r50s1 = float(indexed.loc[("R50", "S1_ORDERLY"), metric])
            r50s2 = float(indexed.loc[("R50", "S2_ACCELERATED"), metric])
            rows.append(
                {
                    **dict(zip(group_keys, key)),
                    "metric": metric,
                    "risk_effect_at_orderly": r50s1 - r0s1,
                    "orderly_buffer_vs_base_under_r50": r50s1 - r50s0,
                    "accelerated_buffer_vs_base_under_r50": r50s2 - r50s0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--climate", required=True, choices=["ssp245", "ssp585"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sections = base.read_query_sections(args.input)
    metrics, storage, national_co2 = build_metrics(sections, args.climate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "province_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    storage.to_csv(args.output_dir / "storage_technology_detail.csv", index=False, encoding="utf-8-sig")
    national_co2.to_csv(args.output_dir / "national_co2_pathways.csv", index=False, encoding="utf-8-sig")
    effects(metrics).to_csv(args.output_dir / "nested_risk_storage_effects.csv", index=False, encoding="utf-8-sig")
    print(f"Processed {len(metrics)} province-scenario-year rows and {len(national_co2)} national CO2 rows.")


if __name__ == "__main__":
    main()
