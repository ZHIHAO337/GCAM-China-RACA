from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
ERA_ROOT = Path(r"E:\GCAM-China-RACA\ERA5-Land\processed\province_daily\parquet_by_year")
NEX_ROOT = Path(r"E:\GCAM-China-RACA\NEX-GDDP-CMIP6\processed\province_daily")
FORMAL = PROJECT / "data" / "formal"
PROCESSED = PROJECT / "data" / "processed"
CONFIG = PROJECT / "config" / "model_assumptions.json"
MODELS = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
SCENARIOS = ["ssp245", "ssp585"]
NODES = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
RETURN_PERIODS = {"R10": 10, "R20": 20, "R50": 50}
METRICS = [
    "heat_degree_days_p95",
    "heat_spell_days_p95",
    "cold_degree_days_p05",
    "cold_spell_days_p05",
    "rx1day_mm",
    "rx5day_mm",
    "heavy_rain_days_50mm",
    "cdd_days",
    "low_wind_spell_days_p10",
    "low_solar_spell_days_p10",
]
HAZARD_WEIGHTS = {
    "heat": {"heat_degree_days_p95": 0.70, "heat_spell_days_p95": 0.30},
    "cold": {"cold_degree_days_p05": 0.70, "cold_spell_days_p05": 0.30},
    "flood": {"rx5day_mm": 0.80, "rx1day_mm": 0.20},
    "drought": {"cdd_days": 1.00},
    "wind": {"low_wind_spell_days_p10": 1.00},
    "solar": {"low_solar_spell_days_p10": 1.00},
}
PARAMETER_HAZARD_WEIGHTS = {
    "delta_peak": {"heat": 0.70, "cold": 0.30},
    "derate_pv": {"solar": 0.75, "heat": 0.25},
    "derate_wind": {"wind": 1.00},
    "derate_hydro": {"drought": 0.85, "flood": 0.15},
    "derate_thermal": {"heat": 0.65, "flood": 0.35},
    "trade_loss": {"flood": 0.427, "wind": 0.160, "cold": 0.226, "heat": 0.187},
}


def max_run(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return 0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return int((changes[1::2] - changes[::2]).max())


def read_many(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )


def standardize_era() -> pd.DataFrame:
    paths = sorted(ERA_ROOT.glob("*.parquet"))
    frame = read_many(
        paths,
        [
            "date",
            "province_code",
            "province_en",
            "province_zh",
            "temperature_2m_max",
            "temperature_2m_min",
            "total_precipitation_sum",
            "surface_solar_radiation_downwards_sum",
            "wind_speed_daily",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["tasmax_c"] = frame["temperature_2m_max"] - 273.15
    frame["tasmin_c"] = frame["temperature_2m_min"] - 273.15
    frame["pr_mm_day"] = frame["total_precipitation_sum"] * 1000.0
    frame["sfcWind_ms"] = frame["wind_speed_daily"].clip(lower=0)
    frame["rsds_wm2"] = (
        frame["surface_solar_radiation_downwards_sum"] / 86400.0
    ).clip(lower=0)
    return frame[
        [
            "date",
            "year",
            "month",
            "province_code",
            "province_en",
            "province_zh",
            "tasmax_c",
            "tasmin_c",
            "pr_mm_day",
            "sfcWind_ms",
            "rsds_wm2",
        ]
    ]


def read_nex(model: str, scenario: str) -> pd.DataFrame:
    paths = sorted((NEX_ROOT / scenario / model).glob("*.parquet"))
    frame = read_many(
        paths,
        [
            "date",
            "province_code",
            "province_en",
            "province_zh",
            "tasmax_c",
            "tasmin_c",
            "pr_mm_day",
            "sfcWind_ms",
            "rsds_wm2",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    return frame


def monthly_thresholds(frame: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    baseline = frame[frame["year"].between(first, last)]
    rows = []
    for (province, month), group in baseline.groupby(["province_code", "month"], sort=False):
        rows.append(
            {
                "province_code": province,
                "month": month,
                "tasmax_p95": group["tasmax_c"].quantile(0.95),
                "tasmin_p05": group["tasmin_c"].quantile(0.05),
                "wind_p10": group["sfcWind_ms"].quantile(0.10),
                "solar_p10": group["rsds_wm2"].quantile(0.10),
            }
        )
    return pd.DataFrame(rows)


def annual_metrics(frame: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(
        thresholds, on=["province_code", "month"], how="left", validate="many_to_one"
    ).sort_values(["province_code", "date"])
    work["heat_excess"] = (work["tasmax_c"] - work["tasmax_p95"]).clip(lower=0)
    work["cold_excess"] = (work["tasmin_p05"] - work["tasmin_c"]).clip(lower=0)
    work["hot_flag"] = work["tasmax_c"] > work["tasmax_p95"]
    work["cold_flag"] = work["tasmin_c"] < work["tasmin_p05"]
    work["low_wind_flag"] = work["sfcWind_ms"] < work["wind_p10"]
    work["low_solar_flag"] = work["rsds_wm2"] < work["solar_p10"]
    work["dry_flag"] = work["pr_mm_day"] < 1.0
    records = []
    for (province, year), group in work.groupby(["province_code", "year"], sort=False):
        group = group.sort_values("date")
        rain = group["pr_mm_day"].to_numpy(dtype=float)
        rx5 = pd.Series(rain).rolling(5, min_periods=5).sum().max()
        records.append(
            {
                "province": province,
                "province_name": group["province_en"].iloc[0],
                "province_zh": group["province_zh"].iloc[0],
                "year": int(year),
                "heat_degree_days_p95": float(group["heat_excess"].sum()),
                "heat_spell_days_p95": max_run(group["hot_flag"].to_numpy()),
                "cold_degree_days_p05": float(group["cold_excess"].sum()),
                "cold_spell_days_p05": max_run(group["cold_flag"].to_numpy()),
                "rx1day_mm": float(np.max(rain)),
                "rx5day_mm": float(rx5),
                "heavy_rain_days_50mm": int(np.count_nonzero(rain >= 50.0)),
                "cdd_days": max_run(group["dry_flag"].to_numpy()),
                "low_wind_spell_days_p10": max_run(group["low_wind_flag"].to_numpy()),
                "low_solar_spell_days_p10": max_run(group["low_solar_flag"].to_numpy()),
            }
        )
    return pd.DataFrame(records)


def to_long_era(annual: pd.DataFrame) -> pd.DataFrame:
    long = annual.melt(
        id_vars=["province", "province_name", "province_zh", "year"],
        value_vars=METRICS,
        var_name="event_metric",
        value_name="annual_value",
    )
    long["spatial_method"] = "31-province pixel-area-weighted daily series"
    long["data_source"] = "ERA5-Land DAILY_AGGR"
    long["data_status"] = "formal_historical_index"
    return long


def empirical_return_levels(era_annual: pd.DataFrame) -> pd.DataFrame:
    long = to_long_era(era_annual)
    rows = []
    for (province, name, zh, metric), group in long.groupby(
        ["province", "province_name", "province_zh", "event_metric"], sort=False
    ):
        values = group["annual_value"].to_numpy(dtype=float)
        for risk, period in RETURN_PERIODS.items():
            rows.append(
                {
                    "province": province,
                    "province_name": name,
                    "province_zh": zh,
                    "event_metric": metric,
                    "risk_level": risk,
                    "return_period_years": period,
                    "return_level": float(np.quantile(values, 1.0 - 1.0 / period)),
                    "sample_years": len(values),
                    "method": "empirical annual-maxima quantile",
                }
            )
    return pd.DataFrame(rows)


def bias_correct_indices(
    era_annual: pd.DataFrame,
    model: str,
    historical: pd.DataFrame,
    future: pd.DataFrame,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    era_overlap = era_annual[era_annual["year"].between(1981, 2014)]
    era_base = era_annual[era_annual["year"].between(1991, 2020)]
    rows = []
    amplification_rows = []
    for province in era_annual["province"].unique():
        era_p = era_overlap[era_overlap["province"] == province]
        base_p = era_base[era_base["province"] == province]
        hist_p = historical[historical["province"] == province]
        fut_p = future[future["province"] == province]
        name = era_p["province_name"].iloc[0]
        zh = era_p["province_zh"].iloc[0]
        for metric in METRICS:
            obs_mean = float(era_p[metric].mean())
            model_mean = float(hist_p[metric].mean())
            correction = obs_mean / max(model_mean, 1e-6)
            correction = float(np.clip(correction, 0.20, 5.00))
            baseline = float(base_p[metric].mean())
            for record in hist_p[["year", metric]].itertuples(index=False):
                rows.append(
                    {
                        "province": province,
                        "province_name": name,
                        "province_zh": zh,
                        "model": model,
                        "scenario": "historical",
                        "year": int(record.year),
                        "event_metric": metric,
                        "annual_value_raw": float(getattr(record, metric)),
                        "bias_factor": correction,
                        "annual_value_corrected": float(getattr(record, metric) * correction),
                        "era5_baseline_1991_2020": baseline,
                        "bias_method": "multiplicative annual-index change-factor correction",
                    }
                )
            corrected_future = fut_p[["year", metric]].copy()
            corrected_future["corrected"] = corrected_future[metric] * correction
            for record in corrected_future.itertuples(index=False):
                rows.append(
                    {
                        "province": province,
                        "province_name": name,
                        "province_zh": zh,
                        "model": model,
                        "scenario": scenario,
                        "year": int(record.year),
                        "event_metric": metric,
                        "annual_value_raw": float(getattr(record, metric)),
                        "bias_factor": correction,
                        "annual_value_corrected": float(record.corrected),
                        "era5_baseline_1991_2020": baseline,
                        "bias_method": "multiplicative annual-index change-factor correction",
                    }
                )
            for node in NODES:
                first = node - 9
                window = corrected_future[corrected_future["year"].between(first, node)]
                corrected_mean = float(window["corrected"].mean())
                raw_mean = float(window[metric].mean())
                amplification = corrected_mean / max(baseline, 1e-6)
                amplification_rows.append(
                    {
                        "province": province,
                        "province_name": name,
                        "province_zh": zh,
                        "model": model,
                        "scenario": scenario,
                        "node_year": node,
                        "window_start": first,
                        "window_end": node,
                        "event_metric": metric,
                        "window_raw_mean": raw_mean,
                        "window_corrected_mean": corrected_mean,
                        "era5_baseline_1991_2020": baseline,
                        "amplification_factor": float(np.clip(amplification, 0.40, 2.50)),
                        "ensemble_status": "member",
                        "data_source": "NEX-GDDP-CMIP6 bias-corrected to ERA5-Land",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(amplification_rows)


def aggregate_hazards(amplification: pd.DataFrame) -> pd.DataFrame:
    member_rows = []
    for keys, group in amplification.groupby(
        ["province", "province_name", "province_zh", "model", "scenario", "node_year"],
        sort=False,
    ):
        values = group.set_index("event_metric")["amplification_factor"].to_dict()
        row = dict(zip(
            ["province", "province_name", "province_zh", "model", "scenario", "node_year"],
            keys,
        ))
        for hazard, weights in HAZARD_WEIGHTS.items():
            row[f"{hazard}_amplification"] = sum(values[metric] * weight for metric, weight in weights.items())
        member_rows.append(row)
    members = pd.DataFrame(member_rows)
    hazard_cols = [f"{hazard}_amplification" for hazard in HAZARD_WEIGHTS]
    records = []
    for keys, group in members.groupby(
        ["province", "province_name", "province_zh", "scenario", "node_year"], sort=False
    ):
        base = dict(zip(
            ["province", "province_name", "province_zh", "scenario", "node_year"], keys
        ))
        for column in hazard_cols:
            values = group[column].to_numpy(dtype=float)
            hazard = column.removesuffix("_amplification")
            records.append(
                {
                    **base,
                    "hazard": hazard,
                    "p10": float(np.quantile(values, 0.10)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.90)),
                    "model_count": int(len(values)),
                }
            )
    return pd.DataFrame(records)


def historical_hazard_multipliers(return_levels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for risk, risk_group in return_levels.groupby("risk_level", sort=False):
        metric_multipliers = {}
        for metric, metric_group in risk_group.groupby("event_metric", sort=False):
            ranked = metric_group.set_index("province")["return_level"].rank(pct=True)
            metric_multipliers[metric] = 0.70 + 0.60 * ranked
        for province in risk_group["province"].unique():
            base = risk_group[risk_group["province"] == province].iloc[0]
            row = {
                "province": province,
                "province_name": base["province_name"],
                "province_zh": base["province_zh"],
                "risk_level": risk,
                "return_period_years": int(base["return_period_years"]),
            }
            for hazard, weights in HAZARD_WEIGHTS.items():
                row[f"{hazard}_historical_multiplier"] = sum(
                    float(metric_multipliers[metric].loc[province]) * weight
                    for metric, weight in weights.items()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def weighted_value(row: pd.Series, parameter: str, suffix: str) -> float:
    return sum(
        row[f"{hazard}_{suffix}"] * weight
        for hazard, weight in PARAMETER_HAZARD_WEIGHTS[parameter].items()
    )


def build_risk_parameters(
    master: pd.DataFrame,
    hazard_ensemble: pd.DataFrame,
    historical_multipliers: pd.DataFrame,
    assumptions: dict,
    scenario: str,
    quantile: str = "median",
) -> pd.DataFrame:
    # Recovering provenance does not turn prescribed impact priors into observations.
    parameter_status = "climate_index_adjusted_prescribed_impacts_not_independent_operational_calibration"
    if assumptions.get("provenance", {}).get("status", "").startswith("reconstructed"):
        parameter_status = "recovered_archival_parameterization_not_independent_empirical_calibration"
    ensemble = hazard_ensemble[hazard_ensemble["scenario"] == scenario].pivot_table(
        index=["province", "node_year"], columns="hazard", values=quantile
    ).reset_index()
    ensemble = ensemble.rename(columns={hazard: f"{hazard}_future_amp" for hazard in HAZARD_WEIGHTS})
    base = master[["province", "province.name", "province_zh", "grid.region"]]
    rows = []
    for risk_level, risk in assumptions["risk_levels"].items():
        if risk_level == "R0":
            for _, province in base.iterrows():
                for year in NODES:
                    rows.append(
                        {
                            "province": province["province"],
                            "province.name": province["province.name"],
                            "province_zh": province["province_zh"],
                            "grid_region": province["grid.region"],
                            "climate_scenario": scenario,
                            "risk_level": risk_level,
                            "return_period_years": 0,
                            "year": year,
                            "climate_amplification": 0.0,
                            "delta_peak": 0.0,
                            "derate_pv": 0.0,
                            "derate_wind": 0.0,
                            "derate_hydro": 0.0,
                            "derate_thermal": 0.0,
                            "trade_loss": 0.0,
                            "equivalent_reserve_margin": assumptions["baseline_reserve_margin"],
                            "parameter_status": parameter_status,
                        }
                    )
            continue
        hist = historical_multipliers[historical_multipliers["risk_level"] == risk_level]
        for _, province in base.iterrows():
            hist_row = hist[hist["province"] == province["province"]].iloc[0]
            for year in NODES:
                future_row = ensemble[
                    (ensemble["province"] == province["province"])
                    & (ensemble["node_year"] == year)
                ].iloc[0]
                combined = pd.Series({**hist_row.to_dict(), **future_row.to_dict()})
                impacts = {}
                for parameter, config_key in [
                    ("delta_peak", "delta_peak"),
                    ("derate_pv", "pv_derate"),
                    ("derate_wind", "wind_derate"),
                    ("derate_hydro", "hydro_derate"),
                    ("derate_thermal", "thermal_derate"),
                    ("trade_loss", "trade_loss"),
                ]:
                    hist_value = weighted_value(combined, parameter, "historical_multiplier")
                    future_value = weighted_value(combined, parameter, "future_amp")
                    cap = 0.50 if parameter == "delta_peak" else 0.85
                    impacts[parameter] = float(
                        np.clip(risk[config_key] * hist_value * future_value, 0.0, cap)
                    )
                climate_amp = float(np.mean([
                    combined[f"{hazard}_future_amp"] for hazard in HAZARD_WEIGHTS
                ]))
                rm = assumptions["baseline_reserve_margin"]
                rows.append(
                    {
                        "province": province["province"],
                        "province.name": province["province.name"],
                        "province_zh": province["province_zh"],
                        "grid_region": province["grid.region"],
                        "climate_scenario": scenario,
                        "risk_level": risk_level,
                        "return_period_years": risk["return_period_years"],
                        "year": year,
                        "climate_amplification": climate_amp,
                        **impacts,
                        "equivalent_reserve_margin": (1 + rm) * (1 + impacts["delta_peak"]) - 1,
                        "parameter_status": parameter_status,
                    }
                )
    return pd.DataFrame(rows)


def grid_sensitivity(risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in risk[risk["risk_level"] != "R0"].itertuples(index=False):
        rows.append(
            {
                "province": row.province,
                "climate_scenario": row.climate_scenario,
                "year": row.year,
                "risk_level": row.risk_level,
                "return_period_years": row.return_period_years,
                "trade_loss_fraction": row.trade_loss,
                "outage_frequency_effect_per_event": 0.0404,
                "outage_duration_effect_per_event": 0.0593,
                "source": "Unequal power outages induced by natural disasters",
                "doi": "https://doi.org/10.1038/s41467-025-64012-x",
                "interpretation": "literature-calibrated sensitivity; not observed line-capacity loss",
                "parameter_status": "formal_sensitivity_parameter_with_direct_empirical_outage_evidence",
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    era_annual: pd.DataFrame,
    cmip_annual: pd.DataFrame,
    amplification: pd.DataFrame,
    hazards: pd.DataFrame,
    risk_main: pd.DataFrame,
) -> dict:
    checks = {
        "era_rows": len(era_annual) == 75 * 31,
        "era_provinces": era_annual["province"].nunique() == 31,
        "cmip_annual_rows": len(cmip_annual) == (34 + 2 * 46) * 5 * 31 * len(METRICS),
        "amplification_rows": len(amplification) == 2 * 5 * 31 * len(NODES) * len(METRICS),
        "hazard_ensemble_rows": len(hazards) == 2 * 31 * len(NODES) * len(HAZARD_WEIGHTS),
        "hazard_five_models": hazards["model_count"].eq(5).all(),
        "risk_rows": len(risk_main) == 31 * 4 * len(NODES),
        "risk_no_missing": not risk_main.isna().any().any(),
        "risk_bounds": (
            risk_main["delta_peak"].between(0, 0.50).all()
            and risk_main[["derate_pv", "derate_wind", "derate_hydro", "derate_thermal", "trade_loss"]]
            .apply(lambda column: column.between(0, 0.85).all())
            .all()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {"status": "verified" if all(checks.values()) else "failed", "checks": checks}


def main() -> None:
    global FORMAL, PROCESSED
    parser = argparse.ArgumentParser()
    parser.add_argument("--assumptions", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, help="Write formal/processed outputs to an independent directory")
    args = parser.parse_args()
    if not args.assumptions.is_file():
        parser.error("Original assumptions file is missing; explicitly select a documented configuration with --assumptions")
    master = pd.read_csv(PROCESSED / "province_master.csv", encoding="utf-8-sig")
    if args.output_root:
        FORMAL = args.output_root / "formal"
        PROCESSED = args.output_root / "processed"
    FORMAL.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    assumptions = json.loads(args.assumptions.read_text(encoding="utf-8"))

    print("STEP=ERA5_LOAD", flush=True)
    era_daily = standardize_era()
    era_thresholds = monthly_thresholds(era_daily, 1991, 2020)
    era_annual = annual_metrics(era_daily, era_thresholds)
    era_long = to_long_era(era_annual)
    return_levels = empirical_return_levels(era_annual)

    all_cmip_annual = []
    all_amplification = []
    for model in MODELS:
        print(f"STEP=CMIP_MODEL_START model={model}", flush=True)
        historical_daily = read_nex(model, "historical")
        model_thresholds = monthly_thresholds(historical_daily, 1981, 2014)
        historical_annual = annual_metrics(historical_daily, model_thresholds)
        for scenario in SCENARIOS:
            print(f"STEP=CMIP_SCENARIO model={model} scenario={scenario}", flush=True)
            future_daily = read_nex(model, scenario)
            future_annual = annual_metrics(future_daily, model_thresholds)
            annual_long, amplification = bias_correct_indices(
                era_annual,
                model,
                historical_annual,
                future_annual,
                scenario,
            )
            if scenario == SCENARIOS[0]:
                all_cmip_annual.append(annual_long[annual_long["scenario"] == "historical"])
            all_cmip_annual.append(annual_long[annual_long["scenario"] == scenario])
            all_amplification.append(amplification)
            del future_daily, future_annual, annual_long, amplification
        del historical_daily, historical_annual

    cmip_annual = pd.concat(all_cmip_annual, ignore_index=True)
    amplification = pd.concat(all_amplification, ignore_index=True)
    hazard_ensemble = aggregate_hazards(amplification)
    historical_multipliers = historical_hazard_multipliers(return_levels)
    risk_ssp245 = build_risk_parameters(
        master, hazard_ensemble, historical_multipliers, assumptions, "ssp245"
    )
    risk_ssp585 = build_risk_parameters(
        master, hazard_ensemble, historical_multipliers, assumptions, "ssp585"
    )
    risk_uncertainty = pd.concat(
        [
            build_risk_parameters(master, hazard_ensemble, historical_multipliers, assumptions, scenario, quantile)
            .assign(ensemble_quantile=quantile)
            for scenario in SCENARIOS
            for quantile in ["p10", "median", "p90"]
        ],
        ignore_index=True,
    )
    grid = pd.concat([grid_sensitivity(risk_ssp245), grid_sensitivity(risk_ssp585)], ignore_index=True)
    validation = validate_outputs(
        era_annual, cmip_annual, amplification, hazard_ensemble, risk_ssp245
    )

    era_long.to_csv(FORMAL / "era5_land_province_indices.csv", index=False, encoding="utf-8-sig")
    era_thresholds.to_csv(FORMAL / "era5_land_monthly_thresholds.csv", index=False, encoding="utf-8-sig")
    return_levels.to_csv(FORMAL / "era5_land_return_levels.csv", index=False, encoding="utf-8-sig")
    cmip_annual.to_parquet(FORMAL / "cmip6_province_annual_indices.parquet", index=False, compression="zstd")
    amplification.to_csv(FORMAL / "cmip6_province_amplification.csv", index=False, encoding="utf-8-sig")
    hazard_ensemble.to_csv(FORMAL / "cmip6_node_hazard_ensemble.csv", index=False, encoding="utf-8-sig")
    historical_multipliers.to_csv(FORMAL / "era5_historical_hazard_multipliers.csv", index=False, encoding="utf-8-sig")
    risk_ssp245.to_csv(FORMAL / "risk_parameters_ssp245.csv", index=False, encoding="utf-8-sig")
    risk_ssp585.to_csv(FORMAL / "risk_parameters_ssp585.csv", index=False, encoding="utf-8-sig")
    risk_uncertainty.to_csv(FORMAL / "risk_parameter_uncertainty.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(FORMAL / "grid_disruption_sensitivity.csv", index=False, encoding="utf-8-sig")
    risk_ssp245.to_csv(PROCESSED / "risk_parameters_long.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "status": validation["status"],
        "assumptions_file": str(args.assumptions.resolve()),
        "assumptions_provenance": assumptions.get("provenance", {}),
        "main_gcam_climate_scenario": "ssp245",
        "sensitivity_climate_scenario": "ssp585",
        "historical_baseline": "ERA5-Land 1991-2020",
        "bias_overlap": "1981-2014",
        "node_windows": {str(node): [node - 9, node] for node in NODES},
        "metrics": METRICS,
        "hazard_weights": HAZARD_WEIGHTS,
        "parameter_hazard_weights": PARAMETER_HAZARD_WEIGHTS,
        "bias_method": "multiplicative annual-index change-factor correction",
        "return_level_method": "empirical quantiles of annual risk indices from 1950-2024",
        "trade_loss_status": "parameterized support-loss sensitivity; independent capacity-loss calibration not established",
        "validation": validation,
    }
    (FORMAL / "climate_processing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if validation["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
