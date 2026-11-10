from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v19"
ANALYSIS = OUT / "long_term_chain"
CLIMATES = ("ssp245", "ssp585")
AMBITIONS = ("CN_NDC7_NZ2060", "CN_NDC10_NZ2060", "IPCC15_ACCELERATED_2050")
STORAGES = ("S0_BASE", "S1_ORDERLY", "S2_ACCELERATED")
REFERENCE = "CN_NDC7_NZ2060"
COMPONENTS = ("required", "nonstorage", "storage", "transfer")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    adequacy = []
    for climate in CLIMATES:
        metrics.append(pd.read_csv(OUT / climate / "processed" / "province_scenario_metrics.csv", encoding="utf-8-sig"))
        adequacy.append(pd.read_csv(OUT / climate / "adequacy" / "province_capacity_adequacy.csv", encoding="utf-8-sig"))
    metrics_frame = pd.concat(metrics, ignore_index=True)
    adequacy_frame = pd.concat(adequacy, ignore_index=True)
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    (OUT / "adequacy").mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(OUT / "processed" / "province_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    adequacy_frame.to_csv(OUT / "adequacy" / "province_capacity_adequacy.csv", index=False, encoding="utf-8-sig")
    return metrics_frame, adequacy_frame


def add_mechanism_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    anchored = [f"{tech}_anchored_capacity_gw" for tech in ("thermal", "hydro", "wind", "solar", "other")]
    result["anchored_capacity_total_gw"] = result[anchored].sum(axis=1)
    result["vre_anchored_capacity_gw"] = result["wind_anchored_capacity_gw"] + result["solar_anchored_capacity_gw"]
    result["vre_anchored_share"] = result["vre_anchored_capacity_gw"] / result["anchored_capacity_total_gw"].replace(0, np.nan)
    result["weather_derating_exposure_gw"] = (
        result["wind_anchored_capacity_gw"] * result["derate_wind"]
        + result["solar_anchored_capacity_gw"] * result["derate_pv"]
        + result["hydro_anchored_capacity_gw"] * result["derate_hydro"]
        + result["thermal_anchored_capacity_gw"] * result["derate_thermal"]
    )
    result["weather_derating_exposure_share"] = result["weather_derating_exposure_gw"] / result[
        "anchored_capacity_total_gw"
    ].replace(0, np.nan)
    result["firm_capacity_retention"] = result["calibrated_nonstorage_firm_capacity_gw"] / result[
        "anchored_capacity_total_gw"
    ].replace(0, np.nan)
    result["transfer_dependence"] = result["available_transfer_support_gw"] / result[
        "required_firm_capacity_gw"
    ].replace(0, np.nan)
    result["storage_power_to_peak"] = result["storage_power_gw"] / result["event_peak_load_gw"].replace(0, np.nan)
    result["storage_energy_to_peak_hours"] = result["storage_energy_gwh"] / result["event_peak_load_gw"].replace(0, np.nan)
    return result


def national_summary(metrics: pd.DataFrame, adequacy: pd.DataFrame) -> pd.DataFrame:
    keys = ["climate_scenario", "ambition_pathway", "risk_level", "storage_pathway", "year"]
    metric_sum = metrics.groupby(keys, as_index=False).agg(
        power_sector_co2_mtc=("co2_mtc", "sum"),
        net_supply_twh=("net_supply_twh", "sum"),
        wind_generation_twh=("wind_generation_twh", "sum"),
        solar_generation_twh=("solar_generation_twh", "sum"),
        thermal_generation_twh=("thermal_generation_twh", "sum"),
        hydro_generation_twh=("hydro_generation_twh", "sum"),
        storage_discharge_twh=("storage_discharge_twh", "sum"),
    )
    sums = adequacy.groupby(keys, as_index=False).agg(
        positive_gap_exposure_gw=("adequacy_gap_gw", "sum"),
        required_firm_capacity_gw=("required_firm_capacity_gw", "sum"),
        nonstorage_firm_capacity_gw=("calibrated_nonstorage_firm_capacity_gw", "sum"),
        storage_firm_capacity_gw=("storage_firm_capacity_gw", "sum"),
        transfer_support_gw=("available_transfer_support_gw", "sum"),
        anchored_capacity_total_gw=("anchored_capacity_total_gw", "sum"),
        vre_anchored_capacity_gw=("vre_anchored_capacity_gw", "sum"),
        weather_derating_exposure_gw=("weather_derating_exposure_gw", "sum"),
        storage_power_gw=("storage_power_gw", "sum"),
        storage_energy_gwh=("storage_energy_gwh", "sum"),
    )
    result = metric_sum.merge(sums, on=keys, validate="one_to_one")
    result["vre_anchored_share"] = result["vre_anchored_capacity_gw"] / result["anchored_capacity_total_gw"]
    result["weather_derating_exposure_share"] = result["weather_derating_exposure_gw"] / result[
        "anchored_capacity_total_gw"
    ]
    result["firm_capacity_retention"] = result["nonstorage_firm_capacity_gw"] / result[
        "anchored_capacity_total_gw"
    ]
    result["storage_power_to_required"] = result["storage_power_gw"] / result["required_firm_capacity_gw"]
    return result


def gap_value(values: dict[str, float]) -> float:
    return max(0.0, values["required"] - values["nonstorage"] - values["storage"] - values["transfer"])


def shapley_decomposition(reference: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    n = len(COMPONENTS)
    contributions = {component: 0.0 for component in COMPONENTS}
    others = set(COMPONENTS)
    for component in COMPONENTS:
        remaining = list(others - {component})
        for size in range(len(remaining) + 1):
            for coalition in itertools.combinations(remaining, size):
                coalition = set(coalition)
                before = {name: (target[name] if name in coalition else reference[name]) for name in COMPONENTS}
                after = dict(before)
                after[component] = target[component]
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                contributions[component] += weight * (gap_value(after) - gap_value(before))
    return contributions


def row_components(row: pd.Series) -> dict[str, float]:
    return {
        "required": float(row["required_firm_capacity_gw"]),
        "nonstorage": float(row["calibrated_nonstorage_firm_capacity_gw"]),
        "storage": float(row["storage_firm_capacity_gw"]),
        "transfer": float(row["available_transfer_support_gw"]),
    }


def ambition_shapley(adequacy: pd.DataFrame) -> pd.DataFrame:
    target = adequacy[
        adequacy["risk_level"].eq("R50") & adequacy["storage_pathway"].eq("S1_ORDERLY")
    ].copy()
    keys = ["climate_scenario", "region", "year"]
    rows = []
    for key, group in target.groupby(keys):
        indexed = group.set_index("ambition_pathway")
        if REFERENCE not in indexed.index:
            continue
        ref = indexed.loc[REFERENCE]
        for ambition in AMBITIONS[1:]:
            if ambition not in indexed.index:
                continue
            hit = indexed.loc[ambition]
            values = shapley_decomposition(row_components(ref), row_components(hit))
            rows.append({
                **dict(zip(keys, key)),
                "ambition_pathway": ambition,
                "reference_pathway": REFERENCE,
                "gap_change_gw": float(hit["adequacy_gap_gw"] - ref["adequacy_gap_gw"]),
                **{f"{name}_contribution_gw": value for name, value in values.items()},
            })
    return pd.DataFrame(rows)


def climate_risk_shapley(adequacy: pd.DataFrame) -> pd.DataFrame:
    target = adequacy[adequacy["storage_pathway"].eq("S1_ORDERLY")].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    rows = []
    for key, group in target.groupby(keys):
        indexed = group.set_index("risk_level")
        if not {"R0", "R50"}.issubset(indexed.index):
            continue
        ref, hit = indexed.loc["R0"], indexed.loc["R50"]
        values = shapley_decomposition(row_components(ref), row_components(hit))
        rows.append({
            **dict(zip(keys, key)),
            "gap_risk_penalty_gw": float(hit["adequacy_gap_gw"] - ref["adequacy_gap_gw"]),
            **{f"{name}_contribution_gw": value for name, value in values.items()},
        })
    return pd.DataFrame(rows)


def storage_effects(adequacy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = adequacy[adequacy["risk_level"].eq("R50")].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    rows = []
    for key, group in target.groupby(keys):
        indexed = group.set_index("storage_pathway")
        if not set(STORAGES).issubset(indexed.index):
            continue
        s0, s1, s2 = indexed.loc["S0_BASE"], indexed.loc["S1_ORDERLY"], indexed.loc["S2_ACCELERATED"]
        rows.append({
            **dict(zip(keys, key)),
            "gap_s0_gw": float(s0["adequacy_gap_gw"]),
            "gap_s1_gw": float(s1["adequacy_gap_gw"]),
            "gap_s2_gw": float(s2["adequacy_gap_gw"]),
            "s0_to_s1_buffer_gw": float(s0["adequacy_gap_gw"] - s1["adequacy_gap_gw"]),
            "s0_to_s2_buffer_gw": float(s0["adequacy_gap_gw"] - s2["adequacy_gap_gw"]),
            "s0_to_s2_relative_buffer": float((s0["adequacy_gap_gw"] - s2["adequacy_gap_gw"]) / s0["adequacy_gap_gw"]) if s0["adequacy_gap_gw"] > 0 else np.nan,
        })
    province = pd.DataFrame(rows)
    reference = province[province["ambition_pathway"].eq(REFERENCE)][
        ["climate_scenario", "region", "year", "s0_to_s2_buffer_gw"]
    ].rename(columns={"s0_to_s2_buffer_gw": "reference_buffer_gw"})
    province = province.merge(reference, on=["climate_scenario", "region", "year"], how="left", validate="many_to_one")
    # This is algebraically equivalent to the interaction definition in the research design.
    province["ambition_storage_interaction_gw"] = province["reference_buffer_gw"] - province["s0_to_s2_buffer_gw"]
    tolerance = 0.05
    province["interaction_class"] = np.select(
        [
            province["ambition_storage_interaction_gw"] < -tolerance,
            province["ambition_storage_interaction_gw"] > tolerance,
        ],
        ["synergy", "weakening"],
        default="independent",
    )
    national = province.groupby(["climate_scenario", "ambition_pathway", "year"], as_index=False).agg(
        gap_s0_gw=("gap_s0_gw", "sum"),
        gap_s1_gw=("gap_s1_gw", "sum"),
        gap_s2_gw=("gap_s2_gw", "sum"),
        s0_to_s1_buffer_gw=("s0_to_s1_buffer_gw", "sum"),
        s0_to_s2_buffer_gw=("s0_to_s2_buffer_gw", "sum"),
        ambition_storage_interaction_gw=("ambition_storage_interaction_gw", "sum"),
        synergy_provinces=("interaction_class", lambda x: int((x == "synergy").sum())),
        independent_provinces=("interaction_class", lambda x: int((x == "independent").sum())),
        weakening_provinces=("interaction_class", lambda x: int((x == "weakening").sum())),
    )
    national["s0_to_s2_relative_buffer"] = national["s0_to_s2_buffer_gw"] / national["gap_s0_gw"].replace(0, np.nan)
    return province, national


def mitigation_resilience_regions(summary: pd.DataFrame) -> pd.DataFrame:
    target = summary[summary["risk_level"].eq("R50")].copy()
    cumulative_rows = []
    cumulative_keys = ["climate_scenario", "ambition_pathway", "storage_pathway"]
    for key, group in target.groupby(cumulative_keys, sort=True):
        group = group.sort_values("year")
        cumulative_rows.append({
            **dict(zip(cumulative_keys, key)),
            "cumulative_power_co2_mtc_year": float(np.trapezoid(group["power_sector_co2_mtc"], group["year"])),
        })
    cumulative = pd.DataFrame(cumulative_rows)
    risk = target[target["year"].eq(2060)][
        ["climate_scenario", "ambition_pathway", "storage_pathway", "positive_gap_exposure_gw"]
    ]
    result = cumulative.merge(risk, on=["climate_scenario", "ambition_pathway", "storage_pathway"], validate="one_to_one")
    ref = result[result["ambition_pathway"].eq(REFERENCE)].rename(columns={
        "cumulative_power_co2_mtc_year": "reference_co2",
        "positive_gap_exposure_gw": "reference_gap",
    })[["climate_scenario", "storage_pathway", "reference_co2", "reference_gap"]]
    result = result.merge(ref, on=["climate_scenario", "storage_pathway"], validate="many_to_one")
    result["mitigation_gain_mtc_year"] = result["reference_co2"] - result["cumulative_power_co2_mtc_year"]
    result["capacity_resilience_gain_gw"] = result["reference_gap"] - result["positive_gap_exposure_gw"]
    result["relation"] = np.select(
        [
            (result["mitigation_gain_mtc_year"] > 0) & (result["capacity_resilience_gain_gw"] > 0),
            (result["mitigation_gain_mtc_year"] > 0) & (result["capacity_resilience_gain_gw"] < 0),
        ],
        ["synergy", "tradeoff"],
        default="reference_or_neutral",
    )
    return result


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics, adequacy = load_inputs()
    adequacy = add_mechanism_metrics(adequacy)
    adequacy.to_csv(OUT / "adequacy" / "province_capacity_adequacy_with_mechanisms.csv", index=False, encoding="utf-8-sig")
    summary = national_summary(metrics, adequacy)
    ambition = ambition_shapley(adequacy)
    climate = climate_risk_shapley(adequacy)
    storage_province, storage_national = storage_effects(adequacy)
    regions = mitigation_resilience_regions(summary)
    outputs = {
        "national_long_term_summary.csv": summary,
        "ambition_gap_shapley_decomposition.csv": ambition,
        "climate_risk_gap_shapley_decomposition.csv": climate,
        "province_storage_adjustment_effects.csv": storage_province,
        "national_storage_adjustment_effects.csv": storage_national,
        "mitigation_capacity_relation.csv": regions,
    }
    for name, frame in outputs.items():
        frame.to_csv(ANALYSIS / name, index=False, encoding="utf-8-sig")

    audit = {
        "metrics_rows": int(len(metrics)),
        "adequacy_rows": int(len(adequacy)),
        "province_count": int(adequacy["region"].nunique()),
        "scenario_count": int(metrics["scenario"].nunique()),
        "ambition_shapley_max_residual": float(
            (ambition["gap_change_gw"] - ambition[[f"{c}_contribution_gw" for c in COMPONENTS]].sum(axis=1)).abs().max()
        ),
        "climate_shapley_max_residual": float(
            (climate["gap_risk_penalty_gw"] - climate[[f"{c}_contribution_gw" for c in COMPONENTS]].sum(axis=1)).abs().max()
        ),
        "missing_mechanism_cells": int(adequacy[[
            "vre_anchored_share", "weather_derating_exposure_share", "firm_capacity_retention",
            "transfer_dependence", "storage_power_to_peak", "storage_energy_to_peak_hours",
        ]].isna().sum().sum()),
    }
    audit["pass"] = bool(
        audit["metrics_rows"] == 5952
        and audit["adequacy_rows"] == 5952
        and audit["province_count"] == 31
        and audit["scenario_count"] == 24
        and audit["ambition_shapley_max_residual"] < 1e-8
        and audit["climate_shapley_max_residual"] < 1e-8
        and audit["missing_mechanism_cells"] == 0
    )
    (ANALYSIS / "long_term_chain_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
