from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


PROJECT = Path(__file__).resolve().parents[1]
ERA_ROOT = Path(
    r"E:\GCAM-China-RACA\ERA5-Land\processed\province_daily\parquet_by_year"
)
HOURS = 168


def load_era5() -> pd.DataFrame:
    columns = [
        "date", "province_code", "province_en", "province_zh",
        "temperature_2m_max", "wind_speed_daily",
        "surface_solar_radiation_downwards_sum",
    ]
    frames = [
        pd.read_parquet(ERA_ROOT / f"era5_land_31prov_daily_{year}.parquet", columns=columns)
        for year in range(1991, 2025)
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["province_code", "date"])


def select_extreme_weeks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = frame[frame["date"].dt.year.le(2020)]
    thresholds = baseline.groupby("province_code").agg(
        heat_p95=("temperature_2m_max", lambda values: values.quantile(0.95)),
        wind_p10=("wind_speed_daily", lambda values: values.quantile(0.10)),
        wind_p90=("wind_speed_daily", lambda values: values.quantile(0.90)),
        solar_p10=("surface_solar_radiation_downwards_sum", lambda values: values.quantile(0.10)),
        solar_p90=("surface_solar_radiation_downwards_sum", lambda values: values.quantile(0.90)),
    ).reset_index()
    scored = frame.merge(thresholds, on="province_code", how="left")
    heat_scale = scored.groupby("province_code")["temperature_2m_max"].transform("std").clip(lower=1)
    wind_scale = scored.groupby("province_code")["wind_speed_daily"].transform("std").clip(lower=0.1)
    solar_scale = scored.groupby("province_code")[
        "surface_solar_radiation_downwards_sum"
    ].transform("std").clip(lower=1)
    scored["daily_compound_score"] = (
        (scored["temperature_2m_max"] - scored["heat_p95"]).clip(lower=0) / heat_scale
        + (scored["wind_p10"] - scored["wind_speed_daily"]).clip(lower=0) / wind_scale
        + (
            scored["solar_p10"] - scored["surface_solar_radiation_downwards_sum"]
        ).clip(lower=0) / solar_scale
    )
    scored["week_score"] = scored.groupby("province_code")[
        "daily_compound_score"
    ].transform(lambda values: values.rolling(7, min_periods=7).sum())
    ends = scored.loc[
        scored.groupby("province_code")["week_score"].idxmax(),
        ["province_code", "date", "week_score"],
    ].rename(columns={"date": "event_end"})
    ends["event_start"] = ends["event_end"] - pd.Timedelta(days=6)
    selected = scored.merge(ends, on="province_code", how="inner")
    selected = selected[
        selected["date"].between(selected["event_start"], selected["event_end"])
    ].copy()
    return selected, ends


def hourly_profiles(week: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    week = week.sort_values("date")
    base_load = np.array([
        0.64, 0.60, 0.58, 0.57, 0.58, 0.63, 0.72, 0.80, 0.86, 0.89, 0.91, 0.92,
        0.90, 0.88, 0.87, 0.89, 0.94, 0.99, 1.00, 0.97, 0.92, 0.85, 0.77, 0.69,
    ])
    load_parts = []
    wind_parts = []
    solar_parts = []
    for row in week.itertuples(index=False):
        heat_multiplier = 1.0 + 0.025 * max(0.0, row.temperature_2m_max - row.heat_p95)
        load_parts.append(base_load * heat_multiplier)
        hours = np.arange(24)
        wind_daily = np.clip((row.wind_speed_daily / max(row.wind_p90, 0.1)) ** 3, 0.02, 1.0)
        wind_parts.append(wind_daily * (1.0 + 0.08 * np.cos(2 * np.pi * (hours - 3) / 24)))
        solar_shape = np.maximum(0.0, np.sin(np.pi * (hours - 6) / 12))
        solar_daily = np.clip(
            row.surface_solar_radiation_downwards_sum / max(row.solar_p90, 1.0), 0.02, 1.0
        )
        solar_parts.append(solar_daily * solar_shape)
    load = np.concatenate(load_parts)
    wind = np.concatenate(wind_parts)
    solar = np.concatenate(solar_parts)
    load /= load.max()
    wind *= 0.25 / max(wind.mean(), 1e-6)
    solar *= 0.17 / max(solar.mean(), 1e-6)
    return load, np.clip(wind, 0, 1), np.clip(solar, 0, 1)


def dispatch(
    load: np.ndarray,
    fixed_supply: np.ndarray,
    storage_power: float,
    storage_energy: float,
    efficiency: float,
    initial_fraction: float,
    terminal_fraction: float,
) -> dict:
    n = len(load)
    charge = np.arange(0, n)
    discharge = np.arange(n, 2 * n)
    soc = np.arange(2 * n, 3 * n)
    unserved = np.arange(3 * n, 4 * n)
    spill = np.arange(4 * n, 5 * n)
    objective = np.zeros(5 * n)
    objective[charge] = 1.0
    objective[discharge] = 1.0
    objective[unserved] = 1e6
    objective[spill] = 0.001
    matrix = lil_matrix((2 * n, 5 * n))
    rhs = np.zeros(2 * n)
    for hour in range(n):
        matrix[hour, discharge[hour]] = 1.0
        matrix[hour, unserved[hour]] = 1.0
        matrix[hour, charge[hour]] = -1.0
        matrix[hour, spill[hour]] = -1.0
        rhs[hour] = load[hour] - fixed_supply[hour]
        row = n + hour
        matrix[row, soc[hour]] = 1.0
        if hour > 0:
            matrix[row, soc[hour - 1]] = -1.0
        matrix[row, charge[hour]] = -np.sqrt(efficiency)
        matrix[row, discharge[hour]] = 1.0 / np.sqrt(efficiency)
        rhs[row] = storage_energy * initial_fraction if hour == 0 else 0.0
    terminal_matrix = lil_matrix((1, 5 * n))
    terminal_matrix[0, soc[-1]] = -1.0
    terminal_rhs = np.array([-storage_energy * terminal_fraction])
    bounds = (
        [(0, storage_power)] * n
        + [(0, storage_power)] * n
        + [(0, storage_energy)] * n
        + [(0, None)] * n
        + [(0, None)] * n
    )
    solution = linprog(
        objective,
        A_ub=terminal_matrix.tocsr(),
        b_ub=terminal_rhs,
        A_eq=matrix.tocsr(),
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    lost = solution.x[unserved]
    return {
        "lole_hours": int((lost > 1e-3).sum()),
        "eens_gwh": float(lost.sum()),
        "max_unserved_gw": float(lost.max()),
        "unserved_energy_share": float(lost.sum() / max(load.sum(), 1e-9)),
        "storage_discharge_gwh": float(solution.x[discharge].sum()),
        "storage_charge_gwh": float(solution.x[charge].sum()),
        "solver_status": solution.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=PROJECT / "outputs/v3/adequacy/province_capacity_adequacy.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "outputs/v3/event_dispatch"
    )
    args = parser.parse_args()
    design = json.loads(
        (PROJECT / "config/reviewer_optimized_design.json").read_text(encoding="utf-8")
    )
    settings = design["representative_event"]
    era5 = load_era5()
    selected, events = select_extreme_weeks(era5)
    adequacy = pd.read_csv(args.input, encoding="utf-8-sig")
    cases = adequacy[
        adequacy["risk_level"].isin(["R20", "R50"])
        & adequacy["year"].isin([2030, 2060])
    ].copy()
    results = []
    profile_rows = []
    for province, week in selected.groupby("province_code"):
        load_shape, wind_shape, solar_shape = hourly_profiles(week)
        for hour in range(HOURS):
            profile_rows.append({
                "province": province,
                "hour": hour,
                "load_shape": load_shape[hour],
                "wind_capacity_factor": wind_shape[hour],
                "solar_capacity_factor": solar_shape[hour],
            })
        for row in cases[cases["region"].eq(province)].itertuples(index=False):
            event_hours = max(1, int(round(row.event_duration_hours)))
            peak_hour = int(load_shape.argmax())
            event_start = max(0, min(HOURS - event_hours, peak_hour - event_hours // 2))
            event_mask = np.zeros(HOURS)
            event_mask[event_start:event_start + event_hours] = 1.0
            load = load_shape * row.normal_peak_load_gw * (
                1.0 + row.delta_peak * event_mask
            )
            fixed = (
                row.thermal * row.thermal_capacity_anchor_factor * 0.90
                * (1.0 - row.derate_thermal * event_mask)
                + row.hydro * row.hydro_capacity_anchor_factor * 0.50
                * (1.0 - row.derate_hydro * event_mask)
                + row.other * row.other_capacity_anchor_factor * 0.50
                + row.wind * row.wind_capacity_anchor_factor * wind_shape
                * (1.0 - row.derate_wind * event_mask)
                + row.solar * row.solar_capacity_anchor_factor * solar_shape
                * (1.0 - row.derate_pv * event_mask)
            ) + row.baseline_transfer_support_gw * (
                1.0 - row.trade_loss * event_mask
            )
            outcome = dispatch(
                load,
                np.asarray(fixed),
                float(row.storage_power_gw),
                float(row.storage_energy_gwh),
                float(settings["round_trip_efficiency"]),
                float(settings["initial_storage_soc_fraction"]),
                float(settings["terminal_storage_soc_fraction"]),
            )
            results.append({
                "climate_scenario": row.climate_scenario,
                "risk_level": row.risk_level,
                "storage_pathway": row.storage_pathway,
                "region": row.region,
                "year": row.year,
                **outcome,
            })
    result = pd.DataFrame(results)
    summary = result.groupby(
        ["climate_scenario", "risk_level", "storage_pathway", "year"], as_index=False
    ).agg(
        provinces_with_lost_load=("lole_hours", lambda values: int((values > 0).sum())),
        median_lole_hours=("lole_hours", "median"),
        p90_lole_hours=("lole_hours", lambda values: values.quantile(0.90)),
        median_eens_gwh=("eens_gwh", "median"),
        p90_eens_gwh=("eens_gwh", lambda values: values.quantile(0.90)),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "province_representative_week_reliability.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "representative_week_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profile_rows).to_csv(args.output_dir / "province_hourly_event_profiles.csv", index=False, encoding="utf-8-sig")
    events.to_csv(args.output_dir / "selected_era5_extreme_weeks.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "hours": HOURS,
        "cases": len(result),
        "method": "linear dispatch with storage state of charge and minimum terminal SOC",
        "scope": settings["dispatch_scope"],
        "profile_status": settings["profile_status"],
    }
    (args.output_dir / "event_dispatch_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
