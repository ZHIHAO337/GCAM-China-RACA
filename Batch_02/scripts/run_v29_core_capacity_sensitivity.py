from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "config" / "reviewer_optimized_design.json"
ADEQUACY_PATH = (
    ROOT
    / "outputs"
    / "v24"
    / "dynamic_climate_uncertainty"
    / "province_capacity_adequacy_uncertainty.parquet"
)
CAPACITY_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "v24"
    / "dynamic_climate_uncertainty"
    / "capacity_uncertainty_summary.csv"
)
OUTPUT = ROOT / "outputs" / "v29_core_capacity_sensitivity"
STORAGE_DETAIL_PATHS = (
    ROOT / "outputs" / "v19" / "ssp245" / "processed" / "storage_technology_detail.csv",
    ROOT / "outputs" / "v19" / "ssp585" / "processed" / "storage_technology_detail.csv",
)

PARAMETERS = [
    "load_factor",
    "reserve_margin",
    "thermal_firm_credit",
    "hydro_firm_credit",
    "wind_firm_credit",
    "solar_firm_credit",
    "storage_availability",
    "storage_duration_multiplier",
    "event_duration_multiplier",
    "transfer_availability_multiplier",
]

CATEGORY_TO_PARAMETER = {
    "thermal": "thermal_firm_credit",
    "hydro": "hydro_firm_credit",
    "wind": "wind_firm_credit",
    "solar": "solar_firm_credit",
}

BASELINE = {
    "load_factor": 0.60,
    "reserve_margin": 0.15,
    "thermal_firm_credit": 0.90,
    "hydro_firm_credit": 0.50,
    "wind_firm_credit": 0.10,
    "solar_firm_credit": 0.05,
    "storage_availability": 0.95,
    "storage_duration_multiplier": 1.0,
    "event_duration_multiplier": 1.0,
    "transfer_availability_multiplier": 1.0,
}

AMBITION_LABELS = {
    "CN_NDC7_NZ2060": "NDC-7",
    "CN_NDC10_NZ2060": "NDC-10",
    "IPCC15_ACCELERATED_2050": "IPCC-1.5C",
}

STORAGE_LABELS = {
    "S0_BASE": "S0",
    "S1_ORDERLY": "S1",
    "S2_ACCELERATED": "S2",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    adequacy = pd.read_parquet(ADEQUACY_PATH)
    adequacy = adequacy[adequacy["ensemble_quantile"].eq("median")].copy()

    storage_frames = [pd.read_csv(path, encoding="utf-8-sig") for path in STORAGE_DETAIL_PATHS]
    storage = pd.concat(storage_frames, ignore_index=True)
    storage = storage[
        [
            "climate_scenario",
            "ambition_pathway",
            "risk_level",
            "storage_pathway",
            "region",
            "year",
            "duration_hours",
            "storage_power_gw",
        ]
    ].copy()
    storage["duration_hours"] = storage["duration_hours"].astype(int)
    storage_power = storage.pivot_table(
        index=[
            "climate_scenario",
            "ambition_pathway",
            "risk_level",
            "storage_pathway",
            "region",
            "year",
        ],
        columns="duration_hours",
        values="storage_power_gw",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    storage_power.columns = [
        f"storage_power_{column}h_gw" if isinstance(column, (int, np.integer)) else column
        for column in storage_power.columns
    ]
    keys = [
        "climate_scenario",
        "ambition_pathway",
        "risk_level",
        "storage_pathway",
        "region",
        "year",
    ]
    adequacy = adequacy.merge(storage_power, on=keys, how="left", validate="one_to_one")
    duration_columns = [f"storage_power_{duration}h_gw" for duration in (2, 4, 8, 10)]
    adequacy[duration_columns] = adequacy[duration_columns].fillna(0.0)

    central_raw_nonstorage = pd.Series(0.0, index=adequacy.index)
    for category, parameter in CATEGORY_TO_PARAMETER.items():
        derate = "derate_pv" if category == "solar" else f"derate_{category}"
        central_raw_nonstorage += (
            adequacy[f"{category}_anchored_capacity_gw"]
            * BASELINE[parameter]
            * (1.0 - adequacy[derate]).clip(0.0, 1.0)
        )
    central_raw_nonstorage += adequacy["other_anchored_capacity_gw"] * 0.50
    adequacy["nonstorage_policy_calibration"] = safe_ratio(
        adequacy["calibrated_nonstorage_firm_capacity_gw"],
        central_raw_nonstorage,
        default=1.0,
    )
    adequacy["transfer_policy_calibration"] = safe_ratio(
        adequacy["available_transfer_support_gw"],
        adequacy["baseline_transfer_support_gw"] * (1.0 - adequacy["trade_loss"]),
        default=0.0,
    )
    return adequacy, storage, design


def ranges_from_design(design: dict) -> dict[str, tuple[float, float]]:
    configured = design["sensitivity"]
    ranges = {name: tuple(float(value) for value in configured[name]) for name in PARAMETERS}
    for name, central in BASELINE.items():
        low, high = ranges[name]
        if not low <= central <= high:
            raise ValueError(f"Central value for {name} lies outside [{low}, {high}]")
    return ranges


def calculate_capacity(frame: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    work = frame.copy()
    work["sampled_normal_peak_load_gw"] = (
        work["normal_peak_load_gw"] * BASELINE["load_factor"] / values["load_factor"]
    )
    work["sampled_event_peak_load_gw"] = work["sampled_normal_peak_load_gw"] * (
        1.0 + work["delta_peak"]
    )
    work["sampled_required_firm_capacity_gw"] = work["sampled_event_peak_load_gw"] * (
        1.0 + values["reserve_margin"]
    )

    nonstorage = pd.Series(0.0, index=work.index)
    for category, parameter in CATEGORY_TO_PARAMETER.items():
        derate = "derate_pv" if category == "solar" else f"derate_{category}"
        nonstorage += (
            work[f"{category}_anchored_capacity_gw"]
            * values[parameter]
            * (1.0 - work[derate]).clip(0.0, 1.0)
        )
    nonstorage += work["other_anchored_capacity_gw"] * 0.50
    work["sampled_nonstorage_firm_capacity_gw"] = (
        nonstorage.clip(lower=0.0) * work["nonstorage_policy_calibration"]
    )

    event_duration = (
        work["event_duration_hours"] * values["event_duration_multiplier"]
    ).to_numpy(dtype=float)
    storage_firm = np.zeros(len(work), dtype=float)
    for duration in (2, 4, 8, 10):
        effective_duration = duration * values["storage_duration_multiplier"]
        duration_credit = np.where(
            event_duration <= 0.0,
            1.0,
            np.minimum(1.0, effective_duration / np.maximum(event_duration, 1e-12)),
        )
        storage_firm += (
            work[f"storage_power_{duration}h_gw"].to_numpy(dtype=float)
            * values["storage_availability"]
            * duration_credit
        )
    work["sampled_storage_firm_capacity_gw"] = storage_firm

    event_transfer = (
        (1.0 - work["trade_loss"])
        * work["transfer_policy_calibration"]
        * values["transfer_availability_multiplier"]
    ).clip(lower=0.0)
    work["sampled_available_transfer_support_gw"] = (
        work["baseline_transfer_support_gw"] * event_transfer
    )
    work["sampled_available_firm_capacity_gw"] = (
        work["sampled_nonstorage_firm_capacity_gw"]
        + work["sampled_storage_firm_capacity_gw"]
        + work["sampled_available_transfer_support_gw"]
    )
    work["sampled_adequacy_gap_gw"] = (
        work["sampled_required_firm_capacity_gw"]
        - work["sampled_available_firm_capacity_gw"]
    ).clip(lower=0.0)
    work["sampled_gap_share_required"] = safe_ratio(
        work["sampled_adequacy_gap_gw"], work["sampled_required_firm_capacity_gw"]
    )
    return work


def summarize_national(work: pd.DataFrame) -> pd.DataFrame:
    target = work[work["risk_level"].eq("R50") & work["year"].isin([2050, 2060])]
    keys = ["climate_scenario", "ambition_pathway", "storage_pathway", "year"]
    return target.groupby(keys, as_index=False).agg(
        national_capacity_gap_gw=("sampled_adequacy_gap_gw", "sum"),
        provinces_with_gap=("sampled_adequacy_gap_gw", lambda values: int((values > 1e-8).sum())),
        median_gap_share=("sampled_gap_share_required", "median"),
    )


def headline_metrics(national: pd.DataFrame) -> dict[str, float]:
    index = national.set_index(
        ["climate_scenario", "ambition_pathway", "storage_pathway", "year"]
    )["national_capacity_gap_gw"]
    climates = sorted(national["climate_scenario"].unique())
    storages = ["S0_BASE", "S1_ORDERLY", "S2_ACCELERATED"]
    ambitions = ["CN_NDC7_NZ2060", "CN_NDC10_NZ2060", "IPCC15_ACCELERATED_2050"]

    metrics: dict[str, float] = {}
    for climate in climates:
        for storage in storages:
            short_climate = climate.upper().replace("SSP", "SSP")
            short_storage = STORAGE_LABELS[storage]
            for year in (2050, 2060):
                metrics[
                    f"ipcc_minus_ndc7_{year}_{short_climate}_{short_storage}_gw"
                ] = float(
                    index.loc[(climate, "IPCC15_ACCELERATED_2050", storage, year)]
                    - index.loc[(climate, "CN_NDC7_NZ2060", storage, year)]
                )
    for year in (2050, 2060):
        values = [
            metrics[f"ipcc_minus_ndc7_{year}_{climate.upper()}_{STORAGE_LABELS[storage]}_gw"]
            for climate in climates
            for storage in storages
        ]
        metrics[f"ipcc_minus_ndc7_{year}_mean_gw"] = float(np.mean(values))

    for ambition in ambitions:
        stage_1: list[float] = []
        stage_2: list[float] = []
        for climate in climates:
            s0 = float(index.loc[(climate, ambition, "S0_BASE", 2060)])
            s1 = float(index.loc[(climate, ambition, "S1_ORDERLY", 2060)])
            s2 = float(index.loc[(climate, ambition, "S2_ACCELERATED", 2060)])
            stage_1.append(s0 - s1)
            stage_2.append(s1 - s2)
        label = AMBITION_LABELS[ambition].replace("-", "_").replace(".", "")
        metrics[f"s0_to_s1_2060_{label}_mean_gw"] = float(np.mean(stage_1))
        metrics[f"s1_to_s2_2060_{label}_mean_gw"] = float(np.mean(stage_2))

    stage_1_all = []
    stage_2_all = []
    for climate in climates:
        for ambition in ambitions:
            s0 = float(index.loc[(climate, ambition, "S0_BASE", 2060)])
            s1 = float(index.loc[(climate, ambition, "S1_ORDERLY", 2060)])
            s2 = float(index.loc[(climate, ambition, "S2_ACCELERATED", 2060)])
            stage_1_all.append(s0 - s1)
            stage_2_all.append(s1 - s2)
    stage_1_mean = float(np.mean(stage_1_all))
    stage_2_mean = float(np.mean(stage_2_all))
    metrics["s0_to_s1_2060_mean_gw"] = stage_1_mean
    metrics["s1_to_s2_2060_mean_gw"] = stage_2_mean
    metrics["s1_to_s2_share_total_2060"] = stage_2_mean / max(stage_1_mean + stage_2_mean, 1e-12)
    return metrics


def scale_lhs(unit: np.ndarray, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    samples = {}
    for column, parameter in enumerate(PARAMETERS):
        low, high = ranges[parameter]
        samples[parameter] = low + unit[:, column] * (high - low)
    return pd.DataFrame(samples)


def partial_rank_correlation(samples: pd.DataFrame, outcome: pd.Series, parameter: str) -> float:
    ranked_x = samples[PARAMETERS].rank(method="average")
    ranked_y = outcome.rank(method="average").to_numpy(dtype=float)
    x = ranked_x[parameter].to_numpy(dtype=float)
    controls = [name for name in PARAMETERS if name != parameter]
    design = np.column_stack(
        [np.ones(len(samples)), ranked_x[controls].to_numpy(dtype=float)]
    )
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    if np.std(x_residual) < 1e-12 or np.std(y_residual) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def summarize_metric_distribution(metric_samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [column for column in metric_samples.columns if column != "sample_id"]:
        values = metric_samples[metric]
        rows.append(
            {
                "metric": metric,
                "n": int(values.notna().sum()),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "positive_fraction": float((values > 0).mean()),
                "negative_fraction": float((values < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_oat(frame: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETERS:
        for setting, value in (
            ("low", ranges[parameter][0]),
            ("central", BASELINE[parameter]),
            ("high", ranges[parameter][1]),
        ):
            values = BASELINE.copy()
            values[parameter] = value
            metrics = headline_metrics(summarize_national(calculate_capacity(frame, values)))
            rows.append(
                {
                    "parameter": parameter,
                    "setting": setting,
                    "parameter_value": value,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    frame, _, design = load_inputs()
    ranges = ranges_from_design(design)
    samples_count = int(design["sensitivity"]["samples"])
    seed = int(design["sensitivity"]["seed"])
    OUTPUT.mkdir(parents=True, exist_ok=True)

    central_work = calculate_capacity(frame, BASELINE)
    component_replication = {
        "required_firm_capacity_gw": float(
            (central_work["sampled_required_firm_capacity_gw"] - central_work["required_firm_capacity_gw"])
            .abs()
            .max()
        ),
        "nonstorage_firm_capacity_gw": float(
            (
                central_work["sampled_nonstorage_firm_capacity_gw"]
                - central_work["calibrated_nonstorage_firm_capacity_gw"]
            )
            .abs()
            .max()
        ),
        "storage_firm_capacity_gw": float(
            (central_work["sampled_storage_firm_capacity_gw"] - central_work["storage_firm_capacity_gw"])
            .abs()
            .max()
        ),
        "available_transfer_support_gw": float(
            (
                central_work["sampled_available_transfer_support_gw"]
                - central_work["available_transfer_support_gw"]
            )
            .abs()
            .max()
        ),
    }
    print("Central component replication:", json.dumps(component_replication, indent=2))
    central_national = summarize_national(central_work)
    published = pd.read_csv(CAPACITY_SUMMARY_PATH, encoding="utf-8-sig")
    published = published[
        published["ensemble_quantile"].eq("median")
        & published["risk_level"].eq("R50")
        & published["year"].isin([2050, 2060])
    ][
        [
            "climate_scenario",
            "ambition_pathway",
            "storage_pathway",
            "year",
            "national_capacity_gap_gw",
        ]
    ]
    replication = central_national.merge(
        published,
        on=["climate_scenario", "ambition_pathway", "storage_pathway", "year"],
        suffixes=("_recalculated", "_published"),
        validate="one_to_one",
    )
    replication["absolute_difference_gw"] = (
        replication["national_capacity_gap_gw_recalculated"]
        - replication["national_capacity_gap_gw_published"]
    ).abs()
    max_replication_difference = float(replication["absolute_difference_gw"].max())
    if max_replication_difference > 1e-6:
        print(
            replication.sort_values("absolute_difference_gw", ascending=False)
            .head(10)
            .to_string(index=False)
        )
        raise RuntimeError(
            "Central-parameter recalculation did not reproduce V29 capacity results; "
            f"maximum absolute difference = {max_replication_difference:.9f} GW"
        )

    sampler = qmc.LatinHypercube(d=len(PARAMETERS), seed=seed)
    samples = scale_lhs(sampler.random(samples_count), ranges)
    samples.insert(0, "sample_id", np.arange(samples_count, dtype=int))

    national_rows = []
    metric_rows = []
    for sample in samples.itertuples(index=False):
        values = {parameter: float(getattr(sample, parameter)) for parameter in PARAMETERS}
        national = summarize_national(calculate_capacity(frame, values))
        national["sample_id"] = int(sample.sample_id)
        national_rows.append(national)
        metric_rows.append({"sample_id": int(sample.sample_id), **headline_metrics(national)})

    national_samples = pd.concat(national_rows, ignore_index=True)
    metric_samples = pd.DataFrame(metric_rows)
    distribution = summarize_metric_distribution(metric_samples)
    oat = build_oat(frame, ranges)

    importance_outcomes = [
        "ipcc_minus_ndc7_2050_mean_gw",
        "ipcc_minus_ndc7_2060_mean_gw",
        "s0_to_s1_2060_mean_gw",
        "s1_to_s2_2060_mean_gw",
        "s1_to_s2_share_total_2060",
    ]
    importance_rows = []
    for outcome in importance_outcomes:
        for parameter in PARAMETERS:
            importance_rows.append(
                {
                    "outcome": outcome,
                    "parameter": parameter,
                    "partial_rank_correlation": partial_rank_correlation(
                        samples, metric_samples[outcome], parameter
                    ),
                }
            )
    importance = pd.DataFrame(importance_rows)
    importance["absolute_partial_rank_correlation"] = importance[
        "partial_rank_correlation"
    ].abs()
    importance["rank_within_outcome"] = importance.groupby("outcome")[
        "absolute_partial_rank_correlation"
    ].rank(method="dense", ascending=False).astype(int)

    sample_sizes = [250, 500, samples_count]
    full = distribution.set_index("metric")[["median", "p05", "p95"]]
    convergence_rows = []
    for size in sample_sizes:
        current = summarize_metric_distribution(metric_samples[metric_samples["sample_id"] < size])
        current = current.set_index("metric")[["median", "p05", "p95"]].join(
            full, rsuffix="_full"
        )
        current["sample_size"] = size
        current["median_absolute_change_from_full"] = (
            current["median"] - current["median_full"]
        ).abs()
        current["p05_absolute_change_from_full"] = (
            current["p05"] - current["p05_full"]
        ).abs()
        current["p95_absolute_change_from_full"] = (
            current["p95"] - current["p95_full"]
        ).abs()
        convergence_rows.append(current.reset_index())
    convergence = pd.concat(convergence_rows, ignore_index=True)

    samples.to_csv(OUTPUT / "lhs_parameter_samples.csv", index=False, encoding="utf-8-sig")
    national_samples.to_parquet(
        OUTPUT / "national_capacity_samples.parquet", index=False, compression="zstd"
    )
    metric_samples.to_csv(
        OUTPUT / "headline_metric_samples.csv", index=False, encoding="utf-8-sig"
    )
    distribution.to_csv(
        OUTPUT / "headline_metric_uncertainty.csv", index=False, encoding="utf-8-sig"
    )
    importance.to_csv(
        OUTPUT / "parameter_importance_prcc.csv", index=False, encoding="utf-8-sig"
    )
    oat.to_csv(OUTPUT / "one_at_a_time_sensitivity.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(
        OUTPUT / "sensitivity_convergence.csv", index=False, encoding="utf-8-sig"
    )
    replication.to_csv(
        OUTPUT / "central_parameter_replication.csv", index=False, encoding="utf-8-sig"
    )

    headline_names = [
        "ipcc_minus_ndc7_2050_mean_gw",
        "ipcc_minus_ndc7_2060_mean_gw",
        "s0_to_s1_2060_mean_gw",
        "s1_to_s2_2060_mean_gw",
        "s1_to_s2_share_total_2060",
    ]
    headline_summary = distribution[distribution["metric"].isin(headline_names)].to_dict(
        orient="records"
    )
    metadata = {
        "model_scope": "final V29 capacity module; median climate ensemble; R50",
        "method": "Latin hypercube global sensitivity with PRCC plus one-at-a-time low/central/high checks",
        "samples": samples_count,
        "seed": seed,
        "parameters": PARAMETERS,
        "parameter_ranges": ranges,
        "central_values": BASELINE,
        "fixed_other_firm_credit": 0.50,
        "central_replication_max_absolute_difference_gw": max_replication_difference,
        "headline_summary": headline_summary,
        "pass": bool(
            max_replication_difference <= 1e-6
            and len(samples) == samples_count
            and national_samples["national_capacity_gap_gw"].ge(0).all()
            and metric_samples.notna().all().all()
        ),
    }
    (OUTPUT / "sensitivity_audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
