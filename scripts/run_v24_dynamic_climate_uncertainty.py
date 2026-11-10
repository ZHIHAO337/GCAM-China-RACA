from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_v19_nested_event_dispatch as dispatch
import run_v20_climate_uncertainty_sensitivity as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v24" / "dynamic_climate_uncertainty"
ALL_MODEL_YEARS = (2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060)
DURATION_SENSITIVITY_YEARS = (2030, 2050, 2060)
EXPECTED_EVENT_ROWS = 117_180


def dynamic_cases_for_window(adequacy: pd.DataFrame, hours: int) -> pd.DataFrame:
    years = ALL_MODEL_YEARS if hours == 168 else DURATION_SENSITIVITY_YEARS
    return adequacy[
        adequacy["risk_level"].eq("R50") & adequacy["year"].isin(years)
    ].copy()


def dynamic_effects(capacity: pd.DataFrame, operation: pd.DataFrame) -> pd.DataFrame:
    capacity = capacity[capacity["risk_level"].eq("R50")].copy()
    operation = operation[operation["risk_level"].eq("R50")].copy()
    cap_index = capacity.set_index([
        "ensemble_quantile", "climate_scenario", "ambition_pathway", "storage_pathway", "year"
    ])
    op_index = operation.set_index([
        "ensemble_quantile", "climate_scenario", "ambition_pathway", "storage_pathway", "year", "window_hours"
    ])
    rows = []
    cells = operation[[
        "ensemble_quantile", "climate_scenario", "ambition_pathway", "year", "window_hours"
    ]].drop_duplicates()
    for cell in cells.itertuples(index=False):
        cap_key = (cell.ensemble_quantile, cell.climate_scenario, cell.ambition_pathway)
        c_s0 = float(cap_index.loc[(*cap_key, "S0_BASE", cell.year), "national_capacity_gap_gw"])
        c_s2 = float(cap_index.loc[(*cap_key, "S2_ACCELERATED", cell.year), "national_capacity_gap_gw"])
        op_key = (*cap_key,)
        o_s0 = float(op_index.loc[(*op_key, "S0_BASE", cell.year, cell.window_hours), "aggregate_worst_event_eens_gwh"])
        o_s2 = float(op_index.loc[(*op_key, "S2_ACCELERATED", cell.year, cell.window_hours), "aggregate_worst_event_eens_gwh"])
        rows.append({
            "ensemble_quantile": cell.ensemble_quantile,
            "climate_scenario": cell.climate_scenario,
            "ambition_pathway": cell.ambition_pathway,
            "year": int(cell.year),
            "window_hours": int(cell.window_hours),
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


def main() -> None:
    dispatch.cases_for_window = dynamic_cases_for_window
    base.dispatch = dispatch
    base.OUTPUT = OUTPUT
    base.build_effects = dynamic_effects
    base.main()

    audit_path = OUTPUT / "climate_uncertainty_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    operation = pd.read_csv(OUTPUT / "operation_uncertainty_summary.csv", encoding="utf-8-sig")
    effects = pd.read_csv(OUTPUT / "storage_effect_uncertainty.csv", encoding="utf-8-sig")
    audit.update({
        "analysis_design": "168h_all_8_nodes;72h_336h_at_2030_2050_2060",
        "expected_event_dispatch_rows": EXPECTED_EVENT_ROWS,
        "operation_year_window_cells": int(operation[["year", "window_hours"]].drop_duplicates().shape[0]),
        "dynamic_storage_effect_rows": int(len(effects)),
    })
    audit["pass"] = bool(
        audit["event_dispatch_rows"] == EXPECTED_EVENT_ROWS
        and audit["province_count"] == 31
        and audit["missing_adequacy_cells"] == 0
        and audit["negative_capacity_gap_rows"] == 0
        and audit["negative_eens_rows"] == 0
        and audit["capacity_quantile_monotonicity_violations"] == 0
        and audit["operation_quantile_monotonicity_violations"] == 0
        and audit["operation_year_window_cells"] == 14
        and audit["dynamic_storage_effect_rows"] == 252
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
