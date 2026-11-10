from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import run_v17_duration_event_dispatch as event_base
from event_supply import event_transfer_support


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v19" / "nested_event_dispatch"
ADEQUACY = ROOT / "outputs" / "v19" / "adequacy" / "province_capacity_adequacy_with_mechanisms.csv"
WINDOW_HOURS = (72, 168, 336)
EVENT_TYPES = event_base.EVENT_TYPES


def best_window(group: pd.DataFrame, score: str, days: int) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    rolling = group[score].rolling(days, min_periods=days).sum()
    end_index = rolling.idxmax()
    event_end = group.loc[end_index, "date"]
    event_start = event_end - pd.Timedelta(days=days - 1)
    return event_start, event_end, float(rolling.loc[end_index])


def select_nested_events(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for province, group in scored.groupby("province_code", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        for event_type in EVENT_TYPES:
            start336, end336, score336 = best_window(group, event_type, 14)
            episode336 = group[group["date"].between(start336, end336)].copy().reset_index(drop=True)
            start168, end168, score168 = best_window(episode336, event_type, 7)
            episode168 = episode336[episode336["date"].between(start168, end168)].copy().reset_index(drop=True)
            start72, end72, score72 = best_window(episode168, event_type, 3)
            episode_id = f"{province}_{event_type}_{start336:%Y%m%d}"
            for hours, start, end, score in (
                (72, start72, end72, score72),
                (168, start168, end168, score168),
                (336, start336, end336, score336),
            ):
                rows.append({
                    "episode_id": episode_id,
                    "province_code": province,
                    "event_type": event_type,
                    "window_hours": hours,
                    "event_start": start,
                    "event_end": end,
                    "parent_336_start": start336,
                    "parent_336_end": end336,
                    "event_score": score,
                })
    return pd.DataFrame(rows)


def dispatch_with_soc(
    load: np.ndarray,
    fixed_supply: np.ndarray,
    storage_power: float,
    storage_energy: float,
    efficiency: float,
    initial_fraction: float,
    terminal_fraction: float,
    *,
    minimum_fraction: float = 0.10,
) -> dict:
    if not (0 <= minimum_fraction <= initial_fraction <= 1 and 0 <= terminal_fraction <= 1):
        raise ValueError("Invalid initial, minimum or terminal SOC fraction")
    if storage_power < 0 or storage_energy < 0 or not (0 < efficiency <= 1):
        raise ValueError("Storage power/energy must be nonnegative and efficiency in (0, 1]")
    if len(load) == 0 or len(load) != len(fixed_supply):
        raise ValueError("Load and supply must have the same nonzero number of hours")
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
        + [(storage_energy * minimum_fraction, storage_energy)] * n
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
    soc_values = solution.x[soc]
    floor = storage_energy * minimum_fraction
    at_floor = np.flatnonzero(soc_values <= floor + max(1e-6, storage_energy * 1e-7))
    return {
        "lole_hours": int((lost > 1e-3).sum()),
        "eens_gwh": float(lost.sum()),
        "max_unserved_gw": float(lost.max()),
        "unserved_energy_share": float(lost.sum() / max(load.sum(), 1e-9)),
        "storage_discharge_gwh": float(solution.x[discharge].sum()),
        "storage_charge_gwh": float(solution.x[charge].sum()),
        "minimum_soc_fraction": float(soc_values.min() / storage_energy) if storage_energy > 0 else np.nan,
        "maximum_soc_fraction": float(soc_values.max() / storage_energy) if storage_energy > 0 else np.nan,
        "terminal_soc_fraction": float(soc_values[-1] / storage_energy) if storage_energy > 0 else np.nan,
        "max_power_balance_residual_gw": float(np.max(np.abs((matrix.tocsr() @ solution.x - rhs)[:n]))),
        "max_soc_balance_residual_gwh": float(np.max(np.abs((matrix.tocsr() @ solution.x - rhs)[n:]))),
        "hours_at_soc_floor": int(len(at_floor)),
        "first_soc_floor_hour": int(at_floor[0]) if len(at_floor) else -1,
        "solver_status": solution.message,
    }


def cases_for_window(adequacy: pd.DataFrame, hours: int) -> pd.DataFrame:
    common = adequacy[adequacy["risk_level"].eq("R50")]
    years = [2030, 2060] if hours == 168 else [2060]
    return common[common["year"].isin(years)].copy()


def run_dispatch(adequacy: pd.DataFrame, scored: pd.DataFrame, events: pd.DataFrame, settings: dict) -> pd.DataFrame:
    results = []
    case_map = {hours: cases_for_window(adequacy, hours) for hours in WINDOW_HOURS}
    for event in events.itertuples(index=False):
        window = scored[
            scored["province_code"].eq(event.province_code)
            & scored["date"].between(event.event_start, event.event_end)
        ].sort_values("date")
        load_shape, wind_shape, solar_shape = event_base.hourly_profiles(window, event.event_type)
        if len(load_shape) != event.window_hours:
            raise RuntimeError(f"Window length mismatch for {event.episode_id}: {len(load_shape)}")
        multiplier = event_base.EVENT_MULTIPLIERS[event.event_type]
        cases = case_map[event.window_hours]
        province_cases = cases[cases["region"].eq(event.province_code)]
        for row in province_cases.itertuples(index=False):
            load = load_shape * row.normal_peak_load_gw * (1.0 + row.delta_peak * multiplier["peak"])
            fixed = (
                row.thermal_anchored_capacity_gw * 0.90 * (1.0 - row.derate_thermal * multiplier["thermal"])
                + row.hydro_anchored_capacity_gw * 0.50 * (1.0 - row.derate_hydro * multiplier["hydro"])
                + row.other_anchored_capacity_gw * 0.50
                + row.wind_anchored_capacity_gw * wind_shape * (1.0 - row.derate_wind * multiplier["wind"])
                + row.solar_anchored_capacity_gw * solar_shape * (1.0 - row.derate_pv * multiplier["pv"])
                + event_transfer_support(row, multiplier["trade"])
            )
            outcome = dispatch_with_soc(
                np.asarray(load),
                np.asarray(fixed),
                float(row.storage_power_gw),
                float(row.storage_energy_gwh),
                float(settings["round_trip_efficiency"]),
                float(settings["initial_storage_soc_fraction"]),
                float(settings["terminal_storage_soc_fraction"]),
                minimum_fraction=float(settings.get("minimum_storage_soc_fraction", 0.10)),
            )
            results.append({
                "climate_scenario": row.climate_scenario,
                "ambition_pathway": row.ambition_pathway,
                "risk_level": row.risk_level,
                "storage_pathway": row.storage_pathway,
                "region": row.region,
                "year": row.year,
                "episode_id": event.episode_id,
                "window_hours": event.window_hours,
                "event_type": event.event_type,
                "event_start": event.event_start,
                "event_end": event.event_end,
                **outcome,
            })
            if len(results) % 1000 == 0:
                print(f"dispatch progress: {len(results):,} rows", flush=True)
    return pd.DataFrame(results)


def summarize(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    province_keys = [
        "climate_scenario", "ambition_pathway", "risk_level", "storage_pathway",
        "region", "year", "window_hours",
    ]
    province = result.groupby(province_keys, as_index=False).agg(
        event_count=("event_type", "nunique"),
        worst_lole_hours=("lole_hours", "max"),
        median_lole_hours=("lole_hours", "median"),
        worst_eens_gwh=("eens_gwh", "max"),
        median_eens_gwh=("eens_gwh", "median"),
        worst_unserved_energy_share=("unserved_energy_share", "max"),
        maximum_hours_at_soc_floor=("hours_at_soc_floor", "max"),
        earliest_soc_floor_hour=("first_soc_floor_hour", lambda x: int(x[x.ge(0)].min()) if x.ge(0).any() else -1),
    )
    summary_keys = [
        "climate_scenario", "ambition_pathway", "risk_level", "storage_pathway",
        "year", "window_hours",
    ]
    national = province.groupby(summary_keys, as_index=False).agg(
        provinces_with_lost_load=("worst_lole_hours", lambda x: int((x > 0).sum())),
        median_worst_lole_hours=("worst_lole_hours", "median"),
        p90_worst_lole_hours=("worst_lole_hours", lambda x: x.quantile(0.90)),
        median_worst_eens_gwh=("worst_eens_gwh", "median"),
        p90_worst_eens_gwh=("worst_eens_gwh", lambda x: x.quantile(0.90)),
        median_worst_unserved_share=("worst_unserved_energy_share", "median"),
        p90_worst_unserved_share=("worst_unserved_energy_share", lambda x: x.quantile(0.90)),
        provinces_reaching_soc_floor=("maximum_hours_at_soc_floor", lambda x: int((x > 0).sum())),
    )
    event_summary = result.groupby(summary_keys + ["event_type"], as_index=False).agg(
        median_eens_gwh=("eens_gwh", "median"),
        p90_eens_gwh=("eens_gwh", lambda x: x.quantile(0.90)),
        median_unserved_share=("unserved_energy_share", "median"),
        provinces_reaching_soc_floor=("hours_at_soc_floor", lambda x: int((x > 0).sum())),
    )
    return province, national, event_summary


def nested_audit(events: pd.DataFrame) -> dict:
    violations = 0
    for _, group in events.groupby(["province_code", "event_type"]):
        indexed = group.set_index("window_hours")
        if not set(WINDOW_HOURS).issubset(indexed.index):
            violations += 1
            continue
        a72, a168, a336 = indexed.loc[72], indexed.loc[168], indexed.loc[336]
        valid = (
            a336.event_start <= a168.event_start <= a168.event_end <= a336.event_end
            and a168.event_start <= a72.event_start <= a72.event_end <= a168.event_end
        )
        violations += int(not valid)
    return {"nested_window_violations": violations}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = json.loads((ROOT / "config" / "reviewer_optimized_design.json").read_text(encoding="utf-8"))
    scored = event_base.build_scores(event_base.load_era5())
    events = select_nested_events(scored)
    adequacy = pd.read_csv(ADEQUACY, encoding="utf-8-sig")
    result = run_dispatch(adequacy, scored, events, design["representative_event"])
    province, national, event_summary = summarize(result)
    result.to_csv(OUTPUT / "province_event_duration_reliability.csv", index=False, encoding="utf-8-sig")
    province.to_csv(OUTPUT / "province_duration_reliability.csv", index=False, encoding="utf-8-sig")
    national.to_csv(OUTPUT / "duration_reliability_summary.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(OUTPUT / "event_type_reliability_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUTPUT / "selected_nested_events.csv", index=False, encoding="utf-8-sig")

    order = {"S0_BASE": 0, "S1_ORDERLY": 1, "S2_ACCELERATED": 2}
    violations = 0
    max_increase = 0.0
    keys = ["climate_scenario", "ambition_pathway", "region", "year", "window_hours", "event_type"]
    for _, group in result.groupby(keys):
        values = group.assign(_order=group["storage_pathway"].map(order)).sort_values("_order")["eens_gwh"].to_numpy()
        increases = np.diff(values)
        violations += int((increases > 1e-8).any())
        if len(increases):
            max_increase = max(max_increase, float(increases.max()))
    audit = {
        "selected_windows": int(len(events)),
        "dispatch_rows": int(len(result)),
        "province_count": int(result["region"].nunique()),
        "event_type_count": int(result["event_type"].nunique()),
        "window_hours": sorted(result["window_hours"].unique().tolist()),
        "missing_cells": int(result.isna().sum().sum() - result["minimum_soc_fraction"].isna().sum()),
        "negative_eens_rows": int((result["eens_gwh"] < -1e-8).sum()),
        "storage_path_nonmonotonic_cases": violations,
        "max_storage_path_increase_gwh": max_increase,
        **nested_audit(events),
    }
    audit["pass"] = bool(
        audit["selected_windows"] == 465
        and audit["dispatch_rows"] == 11160
        and audit["province_count"] == 31
        and audit["event_type_count"] == 5
        and audit["window_hours"] == [72, 168, 336]
        and audit["missing_cells"] == 0
        and audit["negative_eens_rows"] == 0
        and audit["nested_window_violations"] == 0
    )
    (OUTPUT / "nested_event_dispatch_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
