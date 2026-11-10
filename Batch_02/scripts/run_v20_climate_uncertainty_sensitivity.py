from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_v19_nested_event_dispatch as dispatch


ROOT = Path(__file__).resolve().parents[1]
BASE_ADEQUACY = ROOT / "outputs" / "v19" / "adequacy" / "province_capacity_adequacy_with_mechanisms.csv"
RISK = ROOT / "data" / "formal" / "risk_parameter_uncertainty.csv"
OUTPUT = ROOT / "outputs" / "v20" / "climate_uncertainty"
QUANTILES = ("p10", "median", "p90")
DERATE_BY_CATEGORY = {
    "thermal": "derate_thermal",
    "hydro": "derate_hydro",
    "wind": "derate_wind",
    "solar": "derate_pv",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def build_uncertain_adequacy(base: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    keys = ["climate_scenario", "risk_level", "year"]
    risk_keep = [
        "province", *keys, "ensemble_quantile", "climate_amplification",
        "delta_peak", "derate_pv", "derate_wind", "derate_hydro",
        "derate_thermal", "trade_loss", "equivalent_reserve_margin",
    ]
    expanded = base.merge(
        risk[risk_keep],
        left_on=["region", *keys],
        right_on=["province", *keys],
        how="left",
        validate="many_to_many",
        suffixes=("_base", ""),
    )
    if expanded["ensemble_quantile"].isna().any():
        missing = expanded.loc[expanded["ensemble_quantile"].isna(), ["region", *keys]].drop_duplicates()
        raise RuntimeError(f"Missing uncertainty parameters:\n{missing.head()}")

    old_firm = {
        category: expanded[f"{category}_firm_capacity_gw"].copy()
        for category in DERATE_BY_CATEGORY
    }
    raw_firm = pd.Series(0.0, index=expanded.index)
    for category, derate in DERATE_BY_CATEGORY.items():
        anchored = expanded[f"{category}_anchored_capacity_gw"]
        old_derate = expanded[f"{derate}_base"]
        current_firm = old_firm[category]
        availability = safe_ratio(current_firm, anchored * (1.0 - old_derate), default=0.0)
        new_firm = anchored * availability * (1.0 - expanded[derate])
        expanded[f"{category}_firm_capacity_gw"] = new_firm.clip(lower=0.0)
        raw_firm += expanded[f"{category}_firm_capacity_gw"]

    expanded["other_firm_capacity_gw"] = expanded["other_firm_capacity_gw"]
    raw_firm += expanded["other_firm_capacity_gw"]
    # The capacity anchor is already included in every technology-specific term.
    expanded["calibrated_nonstorage_firm_capacity_gw"] = raw_firm

    transfer_scale = safe_ratio(
        expanded["available_transfer_support_gw"],
        expanded["baseline_transfer_support_gw"] * (1.0 - expanded["trade_loss_base"]),
        default=0.0,
    )
    expanded["available_transfer_support_gw"] = (
        expanded["baseline_transfer_support_gw"] * transfer_scale * (1.0 - expanded["trade_loss"])
    ).clip(lower=0.0)
    reserve_requirement_factor = safe_ratio(
        expanded["required_firm_capacity_gw"], expanded["event_peak_load_gw"], default=1.15
    )
    expanded["event_peak_load_gw"] = expanded["normal_peak_load_gw"] * (1.0 + expanded["delta_peak"])
    expanded["required_firm_capacity_gw"] = expanded["event_peak_load_gw"] * reserve_requirement_factor
    expanded["available_firm_capacity_gw"] = (
        expanded["calibrated_nonstorage_firm_capacity_gw"]
        + expanded["storage_firm_capacity_gw"]
        + expanded["available_transfer_support_gw"]
    )
    expanded["adequacy_gap_gw"] = (
        expanded["required_firm_capacity_gw"] - expanded["available_firm_capacity_gw"]
    ).clip(lower=0.0)
    expanded["adequacy_ratio"] = safe_ratio(
        expanded["available_firm_capacity_gw"], expanded["required_firm_capacity_gw"], default=1.0
    )
    expanded["adequacy_gap_share_required"] = safe_ratio(
        expanded["adequacy_gap_gw"], expanded["required_firm_capacity_gw"], default=0.0
    )
    expanded["stress_reserve_margin"] = safe_ratio(
        expanded["available_firm_capacity_gw"], expanded["event_peak_load_gw"], default=1.0
    ) - 1.0
    credit = expanded["portfolio_storage_firm_credit"].clip(lower=0.05)
    expanded["additional_storage_power_required_gw"] = expanded["adequacy_gap_gw"] / credit
    expanded["additional_storage_energy_required_gwh"] = (
        expanded["additional_storage_power_required_gw"]
        * expanded["portfolio_storage_duration_hours"].clip(lower=1.0)
    )
    expanded["risk_adjusted_storage_power_required_gw"] = (
        expanded["storage_power_gw"] + expanded["additional_storage_power_required_gw"]
    )
    expanded["risk_adjusted_storage_energy_required_gwh"] = (
        expanded["storage_energy_gwh"] + expanded["additional_storage_energy_required_gwh"]
    )
    expanded["storage_path_sufficiency_ratio"] = safe_ratio(
        expanded["storage_power_gw"], expanded["risk_adjusted_storage_power_required_gw"], default=1.0
    )
    expanded["shortfall_flag"] = expanded["adequacy_gap_gw"].gt(1e-6)
    expanded["severe_shortfall_flag"] = expanded["adequacy_gap_share_required"].gt(0.10)
    expanded["weather_derating_exposure_gw"] = (
        expanded["wind_anchored_capacity_gw"] * expanded["derate_wind"]
        + expanded["solar_anchored_capacity_gw"] * expanded["derate_pv"]
        + expanded["hydro_anchored_capacity_gw"] * expanded["derate_hydro"]
        + expanded["thermal_anchored_capacity_gw"] * expanded["derate_thermal"]
    )
    expanded["weather_derating_exposure_share"] = safe_ratio(
        expanded["weather_derating_exposure_gw"], expanded["anchored_capacity_total_gw"], default=0.0
    )
    expanded["firm_capacity_retention"] = safe_ratio(
        expanded["calibrated_nonstorage_firm_capacity_gw"] + expanded["storage_firm_capacity_gw"],
        expanded["anchored_capacity_total_gw"] + expanded["storage_power_gw"],
        default=0.0,
    )

    drop = [column for column in expanded.columns if column.endswith("_base")]
    return expanded.drop(columns=["province", *drop])


def summarize_capacity(adequacy: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "ensemble_quantile", "climate_scenario", "ambition_pathway",
        "risk_level", "storage_pathway", "year",
    ]
    return adequacy.groupby(keys, as_index=False).agg(
        national_capacity_gap_gw=("adequacy_gap_gw", "sum"),
        provinces_with_gap=("adequacy_gap_gw", lambda values: int((values > 1e-8).sum())),
        median_gap_share=("adequacy_gap_share_required", "median"),
        p90_gap_share=("adequacy_gap_share_required", lambda values: values.quantile(0.90)),
        median_climate_amplification=("climate_amplification", "median"),
        median_delta_peak=("delta_peak", "median"),
        median_weather_derating=("weather_derating_exposure_share", "median"),
        median_trade_loss=("trade_loss", "median"),
    )


def run_uncertain_dispatch(adequacy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design = json.loads((ROOT / "config" / "reviewer_optimized_design.json").read_text(encoding="utf-8"))
    scored = dispatch.event_base.build_scores(dispatch.event_base.load_era5())
    events = dispatch.select_nested_events(scored)
    all_results = []
    all_province = []
    all_national = []
    for quantile in QUANTILES:
        subset = adequacy[adequacy["ensemble_quantile"].eq(quantile)].copy()
        result = dispatch.run_dispatch(subset, scored, events, design["representative_event"])
        result["ensemble_quantile"] = quantile
        province, national, _ = dispatch.summarize(result)
        province["ensemble_quantile"] = quantile
        national["ensemble_quantile"] = quantile
        all_results.append(result)
        all_province.append(province)
        all_national.append(national)
    return (
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_province, ignore_index=True),
        pd.concat(all_national, ignore_index=True),
    )


def summarize_operation(province: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "ensemble_quantile", "climate_scenario", "ambition_pathway",
        "risk_level", "storage_pathway", "year", "window_hours",
    ]
    return province.groupby(keys, as_index=False).agg(
        aggregate_worst_event_eens_gwh=("worst_eens_gwh", "sum"),
        provinces_with_lost_load=("worst_lole_hours", lambda values: int((values > 0).sum())),
        median_worst_lole_hours=("worst_lole_hours", "median"),
        p90_worst_eens_gwh=("worst_eens_gwh", lambda values: values.quantile(0.90)),
        provinces_reaching_soc_floor=("maximum_hours_at_soc_floor", lambda values: int((values > 0).sum())),
    )


def build_effects(capacity: pd.DataFrame, operation: pd.DataFrame) -> pd.DataFrame:
    cap = capacity[
        capacity["risk_level"].eq("R50") & capacity["year"].eq(2060)
    ].copy()
    op = operation[
        operation["risk_level"].eq("R50") & operation["year"].eq(2060)
    ].copy()
    cap_index = cap.set_index([
        "ensemble_quantile", "climate_scenario", "ambition_pathway", "storage_pathway"
    ])
    op_index = op.set_index([
        "ensemble_quantile", "climate_scenario", "ambition_pathway", "storage_pathway", "window_hours"
    ])
    rows = []
    for quantile in QUANTILES:
        for climate in sorted(cap["climate_scenario"].unique()):
            for ambition in sorted(cap["ambition_pathway"].unique()):
                c_s0 = float(cap_index.loc[(quantile, climate, ambition, "S0_BASE"), "national_capacity_gap_gw"])
                c_s2 = float(cap_index.loc[(quantile, climate, ambition, "S2_ACCELERATED"), "national_capacity_gap_gw"])
                for hours in dispatch.WINDOW_HOURS:
                    o_s0 = float(op_index.loc[(quantile, climate, ambition, "S0_BASE", hours), "aggregate_worst_event_eens_gwh"])
                    o_s2 = float(op_index.loc[(quantile, climate, ambition, "S2_ACCELERATED", hours), "aggregate_worst_event_eens_gwh"])
                    rows.append({
                        "ensemble_quantile": quantile,
                        "climate_scenario": climate,
                        "ambition_pathway": ambition,
                        "window_hours": hours,
                        "capacity_gap_s0_gw": c_s0,
                        "capacity_gap_s2_gw": c_s2,
                        "capacity_gap_reduction_gw": c_s0 - c_s2,
                        "capacity_gap_reduction_pct": 100.0 * (c_s0 - c_s2) / max(c_s0, 1e-9),
                        "eens_s0_gwh": o_s0,
                        "eens_s2_gwh": o_s2,
                        "eens_reduction_gwh": o_s0 - o_s2,
                        "eens_reduction_pct": 100.0 * (o_s0 - o_s2) / max(o_s0, 1e-9),
                    })
    return pd.DataFrame(rows)


def audit_results(
    adequacy: pd.DataFrame,
    capacity: pd.DataFrame,
    event_result: pd.DataFrame,
    operation: pd.DataFrame,
) -> dict:
    order = {"p10": 0, "median": 1, "p90": 2}
    cap_violations = 0
    cap_keys = ["climate_scenario", "ambition_pathway", "risk_level", "storage_pathway", "year"]
    for _, group in capacity.groupby(cap_keys):
        values = group.assign(_order=group["ensemble_quantile"].map(order)).sort_values("_order")["national_capacity_gap_gw"]
        cap_violations += int((np.diff(values.to_numpy()) < -1e-7).any())
    op_violations = 0
    op_keys = ["climate_scenario", "ambition_pathway", "risk_level", "storage_pathway", "year", "window_hours"]
    for _, group in operation.groupby(op_keys):
        values = group.assign(_order=group["ensemble_quantile"].map(order)).sort_values("_order")["aggregate_worst_event_eens_gwh"]
        op_violations += int((np.diff(values.to_numpy()) < -1e-6).any())
    audit = {
        "adequacy_rows": int(len(adequacy)),
        "event_dispatch_rows": int(len(event_result)),
        "province_count": int(adequacy["region"].nunique()),
        "quantiles": sorted(adequacy["ensemble_quantile"].unique().tolist()),
        "missing_adequacy_cells": int(adequacy.isna().sum().sum()),
        "negative_capacity_gap_rows": int((adequacy["adequacy_gap_gw"] < -1e-8).sum()),
        "negative_eens_rows": int((event_result["eens_gwh"] < -1e-8).sum()),
        "capacity_quantile_monotonicity_violations": int(cap_violations),
        "operation_quantile_monotonicity_violations": int(op_violations),
    }
    audit["pass"] = bool(
        audit["province_count"] == 31
        and audit["quantiles"] == ["median", "p10", "p90"]
        and audit["negative_capacity_gap_rows"] == 0
        and audit["negative_eens_rows"] == 0
        and audit["capacity_quantile_monotonicity_violations"] == 0
        and audit["operation_quantile_monotonicity_violations"] == 0
    )
    return audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE_ADEQUACY, encoding="utf-8-sig")
    risk = pd.read_csv(RISK, encoding="utf-8-sig")
    adequacy = build_uncertain_adequacy(base, risk)
    capacity = summarize_capacity(adequacy)
    event_result, province, national = run_uncertain_dispatch(adequacy)
    operation = summarize_operation(province)
    effects = build_effects(capacity, operation)
    audit = audit_results(adequacy, capacity, event_result, operation)

    adequacy.to_parquet(OUTPUT / "province_capacity_adequacy_uncertainty.parquet", index=False, compression="zstd")
    capacity.to_csv(OUTPUT / "capacity_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    event_result.to_parquet(OUTPUT / "province_event_dispatch_uncertainty.parquet", index=False, compression="zstd")
    province.to_csv(OUTPUT / "province_operation_uncertainty.csv", index=False, encoding="utf-8-sig")
    national.to_csv(OUTPUT / "national_operation_uncertainty.csv", index=False, encoding="utf-8-sig")
    operation.to_csv(OUTPUT / "operation_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(OUTPUT / "storage_effect_uncertainty.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "climate_uncertainty_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
