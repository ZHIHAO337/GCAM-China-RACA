from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_v3_representative_event_dispatch as legacy


PROJECT = Path(__file__).resolve().parents[1]
ERA_ROOT = Path(r"E:\GCAM-China-RACA\ERA5-Land\processed\province_daily\parquet_by_year")
WINDOW_HOURS = (72, 168, 336)
EVENT_TYPES = ("compound_heat_low_vre", "heat_load", "low_vre", "cold_derating", "transfer_restriction")
EVENT_MULTIPLIERS = {
    "compound_heat_low_vre": {"peak": 1.0, "pv": 1.0, "wind": 1.0, "hydro": 1.0, "thermal": 1.0, "trade": 1.0},
    "heat_load": {"peak": 1.0, "pv": 0.25, "wind": 0.25, "hydro": 0.25, "thermal": 0.50, "trade": 0.25},
    "low_vre": {"peak": 0.25, "pv": 1.0, "wind": 1.0, "hydro": 0.25, "thermal": 0.25, "trade": 0.25},
    "cold_derating": {"peak": 0.80, "pv": 0.50, "wind": 0.80, "hydro": 0.50, "thermal": 1.0, "trade": 0.50},
    "transfer_restriction": {"peak": 0.25, "pv": 0.25, "wind": 0.50, "hydro": 0.25, "thermal": 0.25, "trade": 1.0},
}


def load_era5() -> pd.DataFrame:
    columns = [
        "date", "province_code", "province_en", "province_zh",
        "temperature_2m_max", "temperature_2m_min", "wind_speed_daily",
        "surface_solar_radiation_downwards_sum", "total_precipitation_sum",
    ]
    frames = [pd.read_parquet(ERA_ROOT / f"era5_land_31prov_daily_{year}.parquet", columns=columns) for year in range(1991, 2025)]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["province_code", "date"])


def build_scores(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["date"].dt.year.le(2020)]
    thresholds = baseline.groupby("province_code").agg(
        heat_p95=("temperature_2m_max", lambda v: v.quantile(0.95)),
        cold_p05=("temperature_2m_min", lambda v: v.quantile(0.05)),
        wind_p10=("wind_speed_daily", lambda v: v.quantile(0.10)),
        wind_p90=("wind_speed_daily", lambda v: v.quantile(0.90)),
        solar_p10=("surface_solar_radiation_downwards_sum", lambda v: v.quantile(0.10)),
        solar_p90=("surface_solar_radiation_downwards_sum", lambda v: v.quantile(0.90)),
        precipitation_p95=("total_precipitation_sum", lambda v: v.quantile(0.95)),
    ).reset_index()
    scored = frame.merge(thresholds, on="province_code", how="left")
    group = scored.groupby("province_code")
    heat_scale = group["temperature_2m_max"].transform("std").clip(lower=1.0)
    cold_scale = group["temperature_2m_min"].transform("std").clip(lower=1.0)
    wind_scale = group["wind_speed_daily"].transform("std").clip(lower=0.1)
    solar_scale = group["surface_solar_radiation_downwards_sum"].transform("std").clip(lower=1.0)
    precipitation_scale = group["total_precipitation_sum"].transform("std").clip(lower=1e-5)
    scored["heat_score"] = ((scored["temperature_2m_max"] - scored["heat_p95"]).clip(lower=0) / heat_scale)
    scored["cold_score"] = ((scored["cold_p05"] - scored["temperature_2m_min"]).clip(lower=0) / cold_scale)
    scored["low_wind_score"] = ((scored["wind_p10"] - scored["wind_speed_daily"]).clip(lower=0) / wind_scale)
    scored["low_solar_score"] = ((scored["solar_p10"] - scored["surface_solar_radiation_downwards_sum"]).clip(lower=0) / solar_scale)
    scored["storm_score"] = (
        (scored["total_precipitation_sum"] - scored["precipitation_p95"]).clip(lower=0) / precipitation_scale
        + (scored["wind_speed_daily"] - scored["wind_p90"]).clip(lower=0) / wind_scale
    )
    # A compound day must contain both heat stress and depressed wind/solar
    # availability. The minimum operator prevents a single large component from
    # being misclassified as a compound event.
    scored["compound_heat_low_vre"] = np.minimum(
        scored["heat_score"], scored["low_wind_score"] + scored["low_solar_score"]
    )
    scored["heat_load"] = scored["heat_score"]
    scored["low_vre"] = scored["low_wind_score"] + scored["low_solar_score"]
    scored["cold_derating"] = scored["cold_score"]
    scored["transfer_restriction"] = scored["storm_score"]
    return scored


def select_events(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for window_hours in WINDOW_HOURS:
        days = window_hours // 24
        work = scored.copy()
        for event_type in EVENT_TYPES:
            work[f"{event_type}_window"] = work.groupby("province_code")[event_type].transform(
                lambda values: values.rolling(days, min_periods=days).sum()
            )
        for province, group in work.groupby("province_code", sort=True):
            occupied: list[pd.Timestamp] = []
            separation = max(14, days)
            for event_type in EVENT_TYPES:
                candidates = group.dropna(subset=[f"{event_type}_window"]).sort_values(f"{event_type}_window", ascending=False)
                chosen = None
                for candidate in candidates.itertuples(index=False):
                    event_end = candidate.date
                    event_start = event_end - pd.Timedelta(days=days - 1)
                    if all(abs((event_start - prior).days) >= separation for prior in occupied):
                        chosen = (event_start, event_end, getattr(candidate, f"{event_type}_window"))
                        break
                if chosen is None:
                    candidate = candidates.iloc[0]
                    chosen = (candidate["date"] - pd.Timedelta(days=days - 1), candidate["date"], candidate[f"{event_type}_window"])
                occupied.append(chosen[0])
                rows.append({
                    "province_code": province,
                    "window_hours": window_hours,
                    "event_type": event_type,
                    "event_start": chosen[0],
                    "event_end": chosen[1],
                    "event_score": float(chosen[2]),
                })
    return pd.DataFrame(rows)


def selected_window(scored: pd.DataFrame, event) -> pd.DataFrame:
    return scored[
        scored["province_code"].eq(event.province_code)
        & scored["date"].between(event.event_start, event.event_end)
    ].copy().sort_values("date")


def hourly_profiles(window: pd.DataFrame, event_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_load = np.array([
        0.64, 0.60, 0.58, 0.57, 0.58, 0.63, 0.72, 0.80, 0.86, 0.89, 0.91, 0.92,
        0.90, 0.88, 0.87, 0.89, 0.94, 0.99, 1.00, 0.97, 0.92, 0.85, 0.77, 0.69,
    ])
    load_parts, wind_parts, solar_parts = [], [], []
    for row in window.itertuples(index=False):
        heat = 0.025 * max(0.0, row.temperature_2m_max - row.heat_p95)
        cold = 0.020 * max(0.0, row.cold_p05 - row.temperature_2m_min)
        weather_multiplier = 1.0 + (heat if event_type in ("heat_load", "compound_heat_low_vre") else 0.0)
        weather_multiplier += cold if event_type == "cold_derating" else 0.0
        load_parts.append(base_load * weather_multiplier)
        hours = np.arange(24)
        wind_daily = np.clip((row.wind_speed_daily / max(row.wind_p90, 0.1)) ** 3, 0.01, 1.0)
        wind_parts.append(wind_daily * (1.0 + 0.08 * np.cos(2 * np.pi * (hours - 3) / 24)))
        solar_shape = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12))
        solar_daily = np.clip(row.surface_solar_radiation_downwards_sum / max(row.solar_p90, 1.0), 0.01, 1.0)
        solar_parts.append(solar_daily * solar_shape)
    return np.concatenate(load_parts), np.clip(np.concatenate(wind_parts), 0, 1), np.clip(np.concatenate(solar_parts), 0, 1)


def cases_for_window(adequacy: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    if window_hours == 168:
        return adequacy[adequacy["risk_level"].isin(["R20", "R50"]) & adequacy["year"].isin([2030, 2060])].copy()
    return adequacy[adequacy["risk_level"].eq("R50") & adequacy["year"].eq(2060)].copy()


def run_dispatch(adequacy: pd.DataFrame, scored: pd.DataFrame, events: pd.DataFrame, settings: dict) -> pd.DataFrame:
    results: list[dict] = []
    cases = {window: cases_for_window(adequacy, window) for window in WINDOW_HOURS}
    baseline_by_window = {}
    key_columns = ["climate_scenario", "risk_level", "region", "year"]
    for window, frame in cases.items():
        baseline = frame[frame["storage_pathway"].eq("S0_BASE")].set_index(key_columns)
        baseline_by_window[window] = baseline
    for event in events.itertuples(index=False):
        window = selected_window(scored, event)
        load_shape, wind_shape, solar_shape = hourly_profiles(window, event.event_type)
        if len(load_shape) != event.window_hours:
            raise RuntimeError(f"Window length mismatch: {event}")
        multiplier = EVENT_MULTIPLIERS[event.event_type]
        for row in cases[event.window_hours][cases[event.window_hours]["region"].eq(event.province_code)].itertuples(index=False):
            base = baseline_by_window[event.window_hours].loc[
                (row.climate_scenario, row.risk_level, row.region, row.year)
            ]
            load = load_shape * base.normal_peak_load_gw * (1.0 + base.delta_peak * multiplier["peak"])
            fixed = (
                base.thermal * base.thermal_capacity_anchor_factor * 0.90 * (1.0 - base.derate_thermal * multiplier["thermal"])
                + base.hydro * base.hydro_capacity_anchor_factor * 0.50 * (1.0 - base.derate_hydro * multiplier["hydro"])
                + base.other * base.other_capacity_anchor_factor * 0.50
                + base.wind * base.wind_capacity_anchor_factor * wind_shape * (1.0 - base.derate_wind * multiplier["wind"])
                + base.solar * base.solar_capacity_anchor_factor * solar_shape * (1.0 - base.derate_pv * multiplier["pv"])
                + base.baseline_transfer_support_gw * (1.0 - base.trade_loss * multiplier["trade"])
            )
            outcome = legacy.dispatch(
                load, np.asarray(fixed), float(row.storage_power_gw), float(row.storage_energy_gwh),
                float(settings["round_trip_efficiency"]), float(settings["initial_storage_soc_fraction"]),
                float(settings["terminal_storage_soc_fraction"]),
            )
            results.append({
                "climate_scenario": row.climate_scenario,
                "risk_level": row.risk_level,
                "storage_pathway": row.storage_pathway,
                "region": row.region,
                "year": row.year,
                "window_hours": event.window_hours,
                "event_type": event.event_type,
                "event_start": event.event_start,
                "event_end": event.event_end,
                **outcome,
            })
    return pd.DataFrame(results)


def summarize(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["climate_scenario", "risk_level", "storage_pathway", "region", "year", "window_hours"]
    province = result.groupby(keys, as_index=False).agg(
        event_count=("event_type", "nunique"),
        worst_lole_hours=("lole_hours", "max"),
        median_lole_hours=("lole_hours", "median"),
        worst_eens_gwh=("eens_gwh", "max"),
        median_eens_gwh=("eens_gwh", "median"),
        worst_unserved_energy_share=("unserved_energy_share", "max"),
    )
    summary_keys = ["climate_scenario", "risk_level", "storage_pathway", "year", "window_hours"]
    summary = province.groupby(summary_keys, as_index=False).agg(
        provinces_with_lost_load=("worst_lole_hours", lambda values: int((values > 0).sum())),
        median_worst_lole_hours=("worst_lole_hours", "median"),
        p90_worst_lole_hours=("worst_lole_hours", lambda values: values.quantile(0.90)),
        median_worst_eens_gwh=("worst_eens_gwh", "median"),
        p90_worst_eens_gwh=("worst_eens_gwh", lambda values: values.quantile(0.90)),
        median_worst_unserved_share=("worst_unserved_energy_share", "median"),
        p90_worst_unserved_share=("worst_unserved_energy_share", lambda values: values.quantile(0.90)),
    )
    event_summary = result.groupby(
        ["climate_scenario", "risk_level", "storage_pathway", "year", "window_hours", "event_type"], as_index=False
    ).agg(
        median_eens_gwh=("eens_gwh", "median"),
        p90_eens_gwh=("eens_gwh", lambda values: values.quantile(0.90)),
        median_unserved_share=("unserved_energy_share", "median"),
    )
    return province, summary, event_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=PROJECT / "outputs/v3/adequacy/province_capacity_adequacy.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/v17/event_dispatch")
    args = parser.parse_args()
    design = json.loads((PROJECT / "config/reviewer_optimized_design.json").read_text(encoding="utf-8"))
    scored = build_scores(load_era5())
    events = select_events(scored)
    adequacy = pd.read_csv(args.input, encoding="utf-8-sig")
    result = run_dispatch(adequacy, scored, events, design["representative_event"])
    province, summary, event_summary = summarize(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "province_event_duration_reliability.csv", index=False, encoding="utf-8-sig")
    province.to_csv(args.output_dir / "province_duration_reliability.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "duration_reliability_summary.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(args.output_dir / "event_type_reliability_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(args.output_dir / "selected_era5_events_by_duration.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "event_types": list(EVENT_TYPES),
        "window_hours": list(WINDOW_HOURS),
        "main_window_hours": 168,
        "sensitivity_windows_hours": [72, 336],
        "selected_windows": int(len(events)),
        "dispatch_cases": int(len(result)),
        "main_scope": "R20/R50, 2030/2060, two climate pathways and three storage pathways",
        "duration_sensitivity_scope": "R50, 2060, two climate pathways and three storage pathways",
        "operational_counterfactual": "matched S0 generation, load and transfer portfolio; only storage power and energy vary across S0/S1/S2",
        "event_multipliers": EVENT_MULTIPLIERS,
        "transfer_event_weather_proxy": "rolling strong-precipitation plus high-wind score",
    }
    (args.output_dir / "duration_event_dispatch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
