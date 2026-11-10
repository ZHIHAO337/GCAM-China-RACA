from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("thermal", "hydro", "wind", "solar", "other")


def wind_solar_policy_floor(year: pd.Series) -> pd.Series:
    """National installed-capacity floors from China's 2030 and 2035 targets."""
    return pd.Series(
        np.select(
            [year.eq(2030), year.ge(2035)],
            [2800.0, 3600.0],
            default=0.0,
        ),
        index=year.index,
    )


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--climate", required=True, choices=["ssp245", "ssp585"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(
        (ROOT / "config" / "reviewer_optimized_design.json").read_text(encoding="utf-8")
    )
    reserve_margin = float(design["baseline_reserve_margin"])

    metrics = pd.read_csv(args.metrics, encoding="utf-8-sig")
    v3_metrics = pd.read_csv(
        ROOT / "outputs" / "v3" / args.climate / "processed" / "province_scenario_metrics.csv",
        encoding="utf-8-sig",
    )
    v3_adequacy = pd.read_csv(
        ROOT / "outputs" / "v3" / args.climate / "adequacy" / "province_capacity_adequacy.csv",
        encoding="utf-8-sig",
    )
    risk = pd.read_csv(
        ROOT / "data" / "v3" / args.climate / "risk_parameters_optimized_interface.csv",
        encoding="utf-8-sig",
    )

    scenario_keys = ["climate_scenario", "risk_level", "storage_pathway", "region", "year"]
    v3_generation = v3_metrics[
        scenario_keys
        + [
            "thermal_generation_twh", "hydro_generation_twh", "wind_generation_twh",
            "solar_generation_twh", "other", "net_supply_twh",
        ]
    ].rename(
        columns={
            "thermal_generation_twh": "v3_thermal_generation_twh",
            "hydro_generation_twh": "v3_hydro_generation_twh",
            "wind_generation_twh": "v3_wind_generation_twh",
            "solar_generation_twh": "v3_solar_generation_twh",
            "other": "v3_other",
            "net_supply_twh": "v3_net_supply_twh",
        }
    )
    anchor_columns = scenario_keys + [
        f"{category}_firm_capacity_gw" for category in CATEGORIES
    ] + [
        f"{category}_anchored_capacity_gw" for category in CATEGORIES
    ] + [
        "nonstorage_calibration_factor", "storage_firm_capacity_gw",
        "baseline_transfer_support_gw", "event_transfer_availability",
        "portfolio_storage_duration_hours", "portfolio_storage_firm_credit",
        "normal_peak_load_gw", "normal_net_supply_twh",
        "delta_peak", "derate_pv", "derate_wind", "derate_hydro", "derate_thermal",
        "trade_loss", "event_duration_hours",
    ]
    reference = v3_adequacy[anchor_columns].copy()
    frame = metrics.merge(v3_generation, on=scenario_keys, how="left").merge(
        reference, on=scenario_keys, how="left", validate="many_to_one"
    )
    if frame["nonstorage_calibration_factor"].isna().any():
        missing = frame.loc[frame["nonstorage_calibration_factor"].isna(), scenario_keys].head()
        raise RuntimeError(f"Missing V3 adequacy bridge rows:\n{missing}")

    source_columns = {
        "thermal": "thermal_generation_twh",
        "hydro": "hydro_generation_twh",
        "wind": "wind_generation_twh",
        "solar": "solar_generation_twh",
        "other": "other",
    }
    for category, source in source_columns.items():
        ratio = safe_ratio(frame[source], frame[f"v3_{source}"]).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.fillna(1.0).clip(lower=0.0, upper=10.0)
        frame[f"{category}_generation_scale_vs_v3"] = ratio
        frame[f"{category}_firm_capacity_gw"] = (
            frame[f"{category}_firm_capacity_gw"] * ratio
        )
        frame[f"{category}_anchored_capacity_gw"] = (
            frame[f"{category}_anchored_capacity_gw"] * ratio
        )

    # The national carbon budget is solved endogenously in GCAM. The near-term
    # wind-plus-solar targets enter here as a binding capacity-floor bridge before
    # adequacy and event dispatch. Incremental capacity is allocated using GCAM's
    # endogenous province and technology shares, preserving the modeled portfolio.
    policy_groups = [
        "climate_scenario", "ambition_pathway", "risk_level", "storage_pathway", "year"
    ]
    frame["wind_solar_capacity_prepolicy_gw"] = (
        frame["wind_anchored_capacity_gw"] + frame["solar_anchored_capacity_gw"]
    )
    frame["national_wind_solar_capacity_prepolicy_gw"] = frame.groupby(policy_groups)[
        "wind_solar_capacity_prepolicy_gw"
    ].transform("sum")
    frame["wind_solar_policy_floor_gw"] = wind_solar_policy_floor(frame["year"])
    required_scale = safe_ratio(
        frame["wind_solar_policy_floor_gw"],
        frame["national_wind_solar_capacity_prepolicy_gw"],
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame["wind_solar_policy_scale"] = np.maximum(1.0, required_scale)
    for category in ("wind", "solar"):
        frame[f"{category}_firm_capacity_gw"] *= frame["wind_solar_policy_scale"]
        frame[f"{category}_anchored_capacity_gw"] *= frame["wind_solar_policy_scale"]
    frame["wind_solar_capacity_policy_adjusted_gw"] = (
        frame["wind_anchored_capacity_gw"] + frame["solar_anchored_capacity_gw"]
    )
    frame["wind_solar_policy_increment_gw"] = (
        frame["wind_solar_capacity_policy_adjusted_gw"]
        - frame["wind_solar_capacity_prepolicy_gw"]
    )
    frame["raw_nonstorage_firm_capacity_gw"] = frame[
        [f"{category}_firm_capacity_gw" for category in CATEGORIES]
    ].sum(axis=1)
    # Technology-specific firm capacities already contain the observed-capacity anchor.
    frame["calibrated_nonstorage_firm_capacity_gw"] = frame["raw_nonstorage_firm_capacity_gw"]

    normal = frame[
        frame["risk_level"].eq("R0") & frame["storage_pathway"].eq("S1_ORDERLY")
    ][["ambition_pathway", "climate_scenario", "region", "year", "net_supply_twh"]].rename(
        columns={"net_supply_twh": "ambition_normal_net_supply_twh"}
    )
    frame = frame.merge(
        normal,
        on=["ambition_pathway", "climate_scenario", "region", "year"],
        how="left",
        validate="many_to_one",
    )
    peak_conversion = safe_ratio(frame["normal_peak_load_gw"], frame["normal_net_supply_twh"])
    frame["normal_peak_load_gw"] = frame["ambition_normal_net_supply_twh"] * peak_conversion
    frame["event_peak_load_gw"] = frame["normal_peak_load_gw"] * (1.0 + frame["delta_peak"])
    frame["baseline_reserve_margin"] = reserve_margin
    frame["required_firm_capacity_gw"] = frame["event_peak_load_gw"] * (1.0 + reserve_margin)

    load_scale = safe_ratio(frame["ambition_normal_net_supply_twh"], frame["normal_net_supply_twh"])
    load_scale = load_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.25, 4.0)
    frame["available_transfer_support_gw"] = (
        frame["baseline_transfer_support_gw"]
        * frame["event_transfer_availability"]
        * load_scale
    )
    expected_requirement = frame["normal_peak_load_gw"] * (1.0 + frame["delta_peak"]) * (1.0 + reserve_margin)
    if not np.allclose(frame["required_firm_capacity_gw"], expected_requirement, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Peak-load shock or reserve margin was applied more than once")
    expected_transfer = (
        frame["baseline_transfer_support_gw"] * frame["event_transfer_availability"] * load_scale
    )
    if not np.allclose(frame["available_transfer_support_gw"], expected_transfer, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Trade-loss adjustment was applied inconsistently")
    frame["available_firm_capacity_gw"] = (
        frame["calibrated_nonstorage_firm_capacity_gw"]
        + frame["storage_firm_capacity_gw"]
        + frame["available_transfer_support_gw"]
    )
    frame["adequacy_gap_gw"] = (
        frame["required_firm_capacity_gw"] - frame["available_firm_capacity_gw"]
    ).clip(lower=0.0)
    frame["adequacy_ratio"] = safe_ratio(
        frame["available_firm_capacity_gw"], frame["required_firm_capacity_gw"]
    )
    frame["adequacy_gap_share_required"] = safe_ratio(
        frame["adequacy_gap_gw"], frame["required_firm_capacity_gw"]
    )
    frame["stress_reserve_margin"] = safe_ratio(
        frame["available_firm_capacity_gw"], frame["event_peak_load_gw"]
    ) - 1.0
    credit = frame["portfolio_storage_firm_credit"].clip(lower=0.05)
    frame["additional_storage_power_required_gw"] = frame["adequacy_gap_gw"] / credit
    frame["additional_storage_energy_required_gwh"] = (
        frame["additional_storage_power_required_gw"]
        * frame["portfolio_storage_duration_hours"].clip(lower=1.0)
    )
    frame["risk_adjusted_storage_power_required_gw"] = (
        frame["storage_power_gw"] + frame["additional_storage_power_required_gw"]
    )
    frame["risk_adjusted_storage_energy_required_gwh"] = (
        frame["storage_energy_gwh"] + frame["additional_storage_energy_required_gwh"]
    )
    frame["storage_path_sufficiency_ratio"] = safe_ratio(
        frame["storage_power_gw"], frame["risk_adjusted_storage_power_required_gw"]
    ).fillna(1.0)
    frame["shortfall_flag"] = frame["adequacy_gap_gw"].gt(1e-6)
    frame["severe_shortfall_flag"] = frame["adequacy_gap_share_required"].gt(0.10)

    effects = []
    group_keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    for key, group in frame.groupby(group_keys):
        indexed = group.set_index(["risk_level", "storage_pathway"])
        if not all(
            item in indexed.index
            for item in [("R0", "S1_ORDERLY"), ("R50", "S0_BASE"), ("R50", "S1_ORDERLY"), ("R50", "S2_ACCELERATED")]
        ):
            continue
        for metric in ("adequacy_gap_gw", "adequacy_ratio", "additional_storage_power_required_gw"):
            r0 = float(indexed.loc[("R0", "S1_ORDERLY"), metric])
            r50s0 = float(indexed.loc[("R50", "S0_BASE"), metric])
            r50s1 = float(indexed.loc[("R50", "S1_ORDERLY"), metric])
            r50s2 = float(indexed.loc[("R50", "S2_ACCELERATED"), metric])
            effects.append(
                {
                    **dict(zip(group_keys, key)),
                    "metric": metric,
                    "risk_penalty_at_orderly": r50s1 - r0,
                    "orderly_storage_buffer": r50s0 - r50s1,
                    "accelerated_storage_buffer": r50s0 - r50s2,
                }
            )

    keep = [
        "scenario", "climate_scenario", "ambition_pathway", "risk_level", "storage_pathway",
        "region", "year", "net_supply_twh", "vre_share", "storage_power_gw", "storage_energy_gwh",
        *[f"{category}_firm_capacity_gw" for category in CATEGORIES],
        *[f"{category}_anchored_capacity_gw" for category in CATEGORIES],
        "storage_firm_capacity_gw", "ambition_normal_net_supply_twh", "normal_peak_load_gw",
        "event_peak_load_gw", "baseline_reserve_margin", "required_firm_capacity_gw", "calibrated_nonstorage_firm_capacity_gw",
        "available_transfer_support_gw", "available_firm_capacity_gw", "adequacy_gap_gw",
        "adequacy_ratio", "adequacy_gap_share_required", "stress_reserve_margin",
        "portfolio_storage_duration_hours", "portfolio_storage_firm_credit",
        "delta_peak", "derate_pv", "derate_wind", "derate_hydro", "derate_thermal",
        "trade_loss", "event_duration_hours", "baseline_transfer_support_gw",
        "wind_solar_capacity_prepolicy_gw", "national_wind_solar_capacity_prepolicy_gw",
        "wind_solar_policy_floor_gw", "wind_solar_policy_scale",
        "wind_solar_capacity_policy_adjusted_gw", "wind_solar_policy_increment_gw",
        "additional_storage_power_required_gw", "additional_storage_energy_required_gwh",
        "risk_adjusted_storage_power_required_gw", "risk_adjusted_storage_energy_required_gwh",
        "storage_path_sufficiency_ratio", "shortfall_flag", "severe_shortfall_flag",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(
        args.output_dir / "province_capacity_adequacy.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(effects).to_csv(
        args.output_dir / "capacity_adequacy_effect_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Built V4 adequacy bridge for {len(frame)} province-scenario-year rows.")


if __name__ == "__main__":
    main()
