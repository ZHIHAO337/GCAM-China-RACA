from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from prepare_v25_dynamic_results import (
    AMBITION_LABEL,
    AMBITION_ORDER,
    BLUE,
    CORAL,
    CORRMAP,
    DIVERGING,
    EVENT_LABEL,
    EVENT_ORDER,
    GOLD,
    GRAY,
    GREEN,
    INK,
    LIGHT,
    PREDICTOR_LABEL,
    PREDICTORS,
    PURPLE,
    SEQUENTIAL,
    TEAL,
    YEARS,
    apply_style,
    draw_heatmap,
    panel,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GCAM_RACA_V29_OUT", ROOT / "outputs" / "v29_three_path_results"))
FIG = OUT / "figures"
TABLE = OUT / "tables"

CAPACITY_FILE = ROOT / "outputs/v24/dynamic_climate_uncertainty/capacity_uncertainty_summary.csv"
OPERATION_FILE = ROOT / "outputs/v24/dynamic_climate_uncertainty/operation_uncertainty_summary.csv"
EVENT_FILE = ROOT / "outputs/v24/dynamic_event_dispatch/province_event_duration_reliability.csv"
DURATION_FILE = ROOT / "outputs/v24/dynamic_event_dispatch/province_duration_reliability.csv"
ADEQUACY_FILE = ROOT / "outputs/v19/adequacy/province_capacity_adequacy_with_mechanisms.csv"
LONG_FILE = ROOT / "outputs/v19/long_term_chain/national_long_term_summary.csv"
DIRECT_FILE = ROOT / "outputs/v24/dynamic_matched_storage_dispatch/matched_storage_direct_effects.csv"
CONFIG_FILE = ROOT / "config/reviewer_optimized_design.json"

PATHS = ["S0_BASE", "S1_ORDERLY", "S2_ACCELERATED"]
PATH_LABEL = {"S0_BASE": "S0", "S1_ORDERLY": "S1", "S2_ACCELERATED": "S2"}
PATH_COLOR = {"S0_BASE": GRAY, "S1_ORDERLY": GOLD, "S2_ACCELERATED": GREEN}
PATH_STYLE = {"S0_BASE": "--", "S1_ORDERLY": "-.", "S2_ACCELERATED": "-"}
TRANSITIONS = ["01", "12", "02"]
TRANSITION_LABEL = {"01": "S0-S1", "12": "S1-S2", "02": "S0-S2"}
TRANSITION_COLOR = {"01": BLUE, "12": CORAL, "02": PURPLE}
KEY_YEARS = [2030, 2050, 2060]
WINDOWS = [72, 168, 336]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    return np.where(np.abs(denominator) > 1e-9, scale * numerator / denominator, np.nan)


def add_stage_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    p0 = out["S0_BASE"]
    p1 = out["S1_ORDERLY"]
    p2 = out["S2_ACCELERATED"]
    out[f"{prefix}_reduction_01"] = p0 - p1
    out[f"{prefix}_reduction_12"] = p1 - p2
    out[f"{prefix}_reduction_02"] = p0 - p2
    for transition in TRANSITIONS:
        out[f"{prefix}_reduction_{transition}_pct_s0"] = safe_ratio(
            out[f"{prefix}_reduction_{transition}"], p0, 100.0
        )
    out[f"{prefix}_reduction_01_pct_stage"] = safe_ratio(out[f"{prefix}_reduction_01"], p0, 100.0)
    out[f"{prefix}_reduction_12_pct_stage"] = safe_ratio(out[f"{prefix}_reduction_12"], p1, 100.0)
    out[f"{prefix}_reduction_02_pct_stage"] = safe_ratio(out[f"{prefix}_reduction_02"], p0, 100.0)
    return out


def build_national_stage_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capacity = read_csv(CAPACITY_FILE)
    capacity = capacity[
        capacity["ensemble_quantile"].eq("median") & capacity["risk_level"].eq("R50")
    ].copy()
    keys = ["climate_scenario", "ambition_pathway", "year"]
    cap = capacity.pivot_table(
        index=keys, columns="storage_pathway", values="national_capacity_gap_gw", aggfunc="first"
    ).reset_index()
    cap = add_stage_metrics(cap, "capacity")

    operation = read_csv(OPERATION_FILE)
    operation = operation[
        operation["ensemble_quantile"].eq("median")
        & operation["risk_level"].eq("R50")
        & operation["window_hours"].eq(168)
    ].copy()
    op = operation.pivot_table(
        index=keys, columns="storage_pathway", values="aggregate_worst_event_eens_gwh", aggfunc="first"
    ).reset_index()
    op = add_stage_metrics(op, "operation")
    stage = cap.merge(op, on=keys, validate="one_to_one", suffixes=("_capacity", "_operation"))

    for transition in TRANSITIONS:
        stage[f"conversion_{transition}"] = safe_ratio(
            stage[f"operation_reduction_{transition}_pct_s0"],
            stage[f"capacity_reduction_{transition}_pct_s0"],
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    powers = config["storage"]["national_power_targets_gw"]
    power_rows = []
    for pathway in PATHS:
        for year, value in powers[pathway].items():
            power_rows.append({"storage_pathway": pathway, "year": int(year), "storage_power_gw": float(value)})
    power = pd.DataFrame(power_rows).pivot(index="year", columns="storage_pathway", values="storage_power_gw").reset_index()
    power["added_power_01_gw"] = power["S1_ORDERLY"] - power["S0_BASE"]
    power["added_power_12_gw"] = power["S2_ACCELERATED"] - power["S1_ORDERLY"]
    stage = stage.merge(power[["year", "added_power_01_gw", "added_power_12_gw"]], on="year", validate="many_to_one")
    for transition in ["01", "12"]:
        stage[f"marginal_capacity_{transition}_gw_per_gw"] = safe_ratio(
            stage[f"capacity_reduction_{transition}"], stage[f"added_power_{transition}_gw"]
        )
        stage[f"marginal_operation_{transition}_gwh_per_gw"] = safe_ratio(
            stage[f"operation_reduction_{transition}"], stage[f"added_power_{transition}_gw"]
        )

    metrics = [
        *PATHS,
        *[f"capacity_reduction_{t}" for t in TRANSITIONS],
        *[f"capacity_reduction_{t}_pct_s0" for t in TRANSITIONS],
        *[f"operation_reduction_{t}" for t in TRANSITIONS],
        *[f"operation_reduction_{t}_pct_s0" for t in TRANSITIONS],
        *[f"conversion_{t}" for t in TRANSITIONS],
        *[f"marginal_capacity_{t}_gw_per_gw" for t in ["01", "12"]],
        *[f"marginal_operation_{t}_gwh_per_gw" for t in ["01", "12"]],
    ]
    # Capacity and operation share pathway names after merge; make explicit aliases.
    for pathway in PATHS:
        stage[f"capacity_{pathway}"] = stage[f"{pathway}_capacity"]
        stage[f"operation_{pathway}"] = stage[f"{pathway}_operation"]
    summary_metrics = [
        *[f"capacity_{p}" for p in PATHS],
        *[f"operation_{p}" for p in PATHS],
        *[f"capacity_reduction_{t}" for t in TRANSITIONS],
        *[f"capacity_reduction_{t}_pct_s0" for t in TRANSITIONS],
        *[f"operation_reduction_{t}" for t in TRANSITIONS],
        *[f"operation_reduction_{t}_pct_s0" for t in TRANSITIONS],
        *[f"conversion_{t}" for t in TRANSITIONS],
        *[f"marginal_capacity_{t}_gw_per_gw" for t in ["01", "12"]],
        *[f"marginal_operation_{t}_gwh_per_gw" for t in ["01", "12"]],
    ]
    trajectory = stage.groupby("year", as_index=False)[summary_metrics].agg(["median", "min", "max"])
    trajectory.columns = ["_".join([str(value) for value in column if value]).rstrip("_") for column in trajectory.columns]
    trajectory = trajectory.rename(columns={"year_": "year"})
    return stage, trajectory, power


def build_ambition_robustness() -> pd.DataFrame:
    capacity = read_csv(CAPACITY_FILE)
    capacity = capacity[
        capacity["ensemble_quantile"].eq("median") & capacity["risk_level"].eq("R50")
    ].copy()
    operation = read_csv(OPERATION_FILE)
    operation = operation[
        operation["ensemble_quantile"].eq("median")
        & operation["risk_level"].eq("R50")
        & operation["window_hours"].eq(168)
    ].copy()
    keys = ["climate_scenario", "storage_pathway", "year"]
    cap = capacity.pivot_table(index=keys, columns="ambition_pathway", values="national_capacity_gap_gw").reset_index()
    op = operation.pivot_table(index=keys, columns="ambition_pathway", values="aggregate_worst_event_eens_gwh").reset_index()
    out = cap.merge(op, on=keys, suffixes=("_cap", "_op"), validate="one_to_one")
    for comparison, target in [("NDC10", "CN_NDC10_NZ2060"), ("IPCC15", "IPCC15_ACCELERATED_2050")]:
        out[f"capacity_delta_{comparison}_vs_NDC7_gw"] = out[f"{target}_cap"] - out["CN_NDC7_NZ2060_cap"]
        out[f"operation_delta_{comparison}_vs_NDC7_gwh"] = out[f"{target}_op"] - out["CN_NDC7_NZ2060_op"]
    return out


def build_province_stage_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duration = read_csv(DURATION_FILE)
    duration = duration[duration["risk_level"].eq("R50") & duration["window_hours"].eq(168)].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    province = duration.pivot_table(index=keys, columns="storage_pathway", values="worst_eens_gwh").reset_index()
    province = add_stage_metrics(province, "operation")
    province["s1_share_total_benefit_pct"] = safe_ratio(
        province["operation_reduction_01"], province["operation_reduction_02"], 100.0
    )
    province["s2_increment_share_total_benefit_pct"] = safe_ratio(
        province["operation_reduction_12"], province["operation_reduction_02"], 100.0
    )

    adequacy = read_csv(ADEQUACY_FILE)
    adequacy = adequacy[adequacy["risk_level"].eq("R50") & adequacy["storage_pathway"].isin(["S1_ORDERLY", "S2_ACCELERATED"])].copy()
    correlation_rows = []
    for pathway in ["S1_ORDERLY", "S2_ACCELERATED"]:
        path_adequacy = adequacy[adequacy["storage_pathway"].eq(pathway)]
        merged = province.merge(path_adequacy[keys + PREDICTORS], on=keys, validate="one_to_one")
        for (climate, ambition, year), group in merged.groupby(["climate_scenario", "ambition_pathway", "year"]):
            for predictor in PREDICTORS:
                correlation_rows.append({
                    "climate_scenario": climate,
                    "ambition_pathway": ambition,
                    "year": int(year),
                    "storage_pathway": pathway,
                    "predictor": predictor,
                    "spearman_rho": float(group[[predictor, pathway]].corr(method="spearman").iloc[0, 1]),
                })
    correlations = pd.DataFrame(correlation_rows)

    endpoint = province[province["year"].eq(2060)].groupby("region", as_index=False).agg(
        residual_s1_gwh=("S1_ORDERLY", "median"),
        residual_s2_gwh=("S2_ACCELERATED", "median"),
        benefit_01_pct=("operation_reduction_01_pct_s0", "median"),
        benefit_12_pct=("operation_reduction_12_pct_s0", "median"),
        benefit_02_pct=("operation_reduction_02_pct_s0", "median"),
        s1_share_pct=("s1_share_total_benefit_pct", "median"),
    ).sort_values("residual_s2_gwh", ascending=False)
    return province, correlations, endpoint


def build_event_stage_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event = read_csv(EVENT_FILE)
    event = event[event["risk_level"].eq("R50")].copy()
    group_keys = ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type", "storage_pathway"]
    national = event.groupby(group_keys, as_index=False)["eens_gwh"].sum()
    pivot_keys = ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type"]
    event_stage = national.pivot_table(index=pivot_keys, columns="storage_pathway", values="eens_gwh").reset_index()
    event_stage = add_stage_metrics(event_stage, "operation")

    floor = event[event["storage_pathway"].isin(["S1_ORDERLY", "S2_ACCELERATED"])].copy()
    floor["reaches_floor"] = floor["hours_at_soc_floor"].gt(0).astype(int)
    floor = floor.groupby(
        ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type", "storage_pathway"],
        as_index=False,
    )["reaches_floor"].sum()
    floor["soc_floor_share_pct"] = 100.0 * floor["reaches_floor"] / 31.0

    total_keys = ["climate_scenario", "ambition_pathway", "region", "year", "window_hours", "event_type"]
    total = event.pivot_table(index=total_keys, columns="storage_pathway", values="eens_gwh").reset_index()
    total["total_effect_gwh"] = total["S0_BASE"] - total["S2_ACCELERATED"]
    direct = read_csv(DIRECT_FILE)
    matched = total.merge(direct[total_keys + ["direct_storage_eens_reduction_gwh"]], on=total_keys, validate="one_to_one")
    matched["portfolio_substitution_gwh"] = matched["total_effect_gwh"] - matched["direct_storage_eens_reduction_gwh"]
    matched["negative_total_effect"] = matched["total_effect_gwh"].lt(-1e-8)
    return event_stage, floor, matched


def build_duration_stage_data() -> pd.DataFrame:
    operation = read_csv(OPERATION_FILE)
    operation = operation[
        operation["ensemble_quantile"].eq("median")
        & operation["risk_level"].eq("R50")
        & operation["year"].isin(KEY_YEARS)
    ].copy()
    keys = ["climate_scenario", "ambition_pathway", "year", "window_hours"]
    duration = operation.pivot_table(
        index=keys, columns="storage_pathway", values="aggregate_worst_event_eens_gwh", aggfunc="first"
    ).reset_index()
    return add_stage_metrics(duration, "operation")


def build_climate_stage_sensitivity() -> pd.DataFrame:
    capacity = read_csv(CAPACITY_FILE)
    capacity = capacity[capacity["risk_level"].eq("R50") & capacity["year"].eq(2060)].copy()
    cap_keys = ["ensemble_quantile", "climate_scenario", "ambition_pathway"]
    cap = capacity.pivot_table(index=cap_keys, columns="storage_pathway", values="national_capacity_gap_gw").reset_index()
    cap = add_stage_metrics(cap, "capacity")

    operation = read_csv(OPERATION_FILE)
    operation = operation[operation["risk_level"].eq("R50") & operation["year"].eq(2060)].copy()
    op_keys = ["ensemble_quantile", "climate_scenario", "ambition_pathway", "window_hours"]
    op = operation.pivot_table(index=op_keys, columns="storage_pathway", values="aggregate_worst_event_eens_gwh").reset_index()
    op = add_stage_metrics(op, "operation")
    out = op.merge(cap, on=cap_keys, suffixes=("_op", "_cap"), validate="many_to_one")
    for transition in TRANSITIONS:
        out[f"conversion_{transition}"] = safe_ratio(
            out[f"operation_reduction_{transition}_pct_s0"],
            out[f"capacity_reduction_{transition}_pct_s0"],
        )
    return out


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def figure_4(stage: pd.DataFrame, trajectory: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    for pathway in PATHS:
        column = f"capacity_{pathway}"
        ax.plot(
            trajectory["year"], trajectory[f"{column}_median"], color=PATH_COLOR[pathway],
            lw=1.45, marker="o", ms=2.8, ls=PATH_STYLE[pathway], label=PATH_LABEL[pathway],
        )
        ax.fill_between(
            trajectory["year"], trajectory[f"{column}_min"], trajectory[f"{column}_max"],
            color=PATH_COLOR[pathway], alpha=0.10, linewidth=0,
        )
    ax.set_ylabel("Positive capacity gap (GW)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    ax.legend(ncol=3, loc="upper left", columnspacing=0.8)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for pathway in PATHS:
        column = f"operation_{pathway}"
        ax.plot(
            trajectory["year"], trajectory[f"{column}_median"] / 1000.0, color=PATH_COLOR[pathway],
            lw=1.45, marker="o", ms=2.8, ls=PATH_STYLE[pathway], label=PATH_LABEL[pathway],
        )
        ax.fill_between(
            trajectory["year"], trajectory[f"{column}_min"] / 1000.0,
            trajectory[f"{column}_max"] / 1000.0, color=PATH_COLOR[pathway], alpha=0.10, linewidth=0,
        )
    ax.set_ylabel("Worst-event ENS, 168 h (TWh)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    visible = trajectory[trajectory["year"].gt(2025)]
    for transition, linestyle in [("01", "-"), ("12", "--")]:
        ax.plot(
            visible["year"], visible[f"capacity_reduction_{transition}_pct_s0_median"],
            color=BLUE, marker="s", ms=2.8, lw=1.35, ls=linestyle,
            label=f"Capacity {TRANSITION_LABEL[transition]}",
        )
        ax.plot(
            visible["year"], visible[f"operation_reduction_{transition}_pct_s0_median"],
            color=CORAL, marker="o", ms=2.8, lw=1.35, ls=linestyle,
            label=f"Operation {TRANSITION_LABEL[transition]}",
        )
    ax.axhline(0, color=INK, lw=0.45)
    ax.set_ylabel("Incremental reduction vs S0 (%)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[1::2], YEARS[1::2])
    ax.legend(ncol=2, loc="upper left", columnspacing=0.7, handlelength=1.6)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    heat_rows = []
    for ambition in AMBITION_ORDER:
        for transition in ["01", "12"]:
            row = stage[stage["ambition_pathway"].eq(ambition)].groupby("year")[f"conversion_{transition}"].median()
            row.name = f"{AMBITION_LABEL[ambition]} {TRANSITION_LABEL[transition]}"
            heat_rows.append(row)
    matrix = pd.DataFrame(heat_rows).reindex(columns=YEARS[1:])
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    draw_heatmap(
        fig, ax, matrix, SEQUENTIAL, "Capacity-to-operation conversion",
        norm=Normalize(0, max(2.2, float(np.nanquantile(finite, 0.98)))), fmt=".2f",
    )
    ax.set_xlabel("GCAM node")
    ax.tick_params(axis="y", labelsize=4.6)
    panel(ax, "d")
    save_figure(fig, "figure_4_dynamic_three_path_mediation")


def figure_6(stage: pd.DataFrame, province: pd.DataFrame, correlations: pd.DataFrame, endpoint: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18], hspace=0.50, wspace=0.44)

    ax = fig.add_subplot(gs[0, 0])
    plot = stage[stage["year"].gt(2025)].copy()
    year_norm = Normalize(2030, 2060)
    for transition, marker in [("01", "o"), ("12", "s")]:
        ax.scatter(
            plot[f"capacity_reduction_{transition}_pct_s0"],
            plot[f"operation_reduction_{transition}_pct_s0"],
            c=plot["year"], cmap=SEQUENTIAL, norm=year_norm, marker=marker,
            s=22, alpha=0.78, edgecolor="white", linewidth=0.35,
            label=TRANSITION_LABEL[transition],
        )
    high = max(
        float(plot[["capacity_reduction_01_pct_s0", "capacity_reduction_12_pct_s0"]].max().max()),
        float(plot[["operation_reduction_01_pct_s0", "operation_reduction_12_pct_s0"]].max().max()),
    ) * 1.06
    low = min(
        0.0,
        float(plot[["capacity_reduction_01_pct_s0", "capacity_reduction_12_pct_s0"]].min().min()),
        float(plot[["operation_reduction_01_pct_s0", "operation_reduction_12_pct_s0"]].min().min()),
    )
    ax.plot([low, high], [low, high], color=LIGHT, lw=0.8, zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Capacity-gap reduction vs S0 (%)")
    ax.set_ylabel("168 h ENS reduction vs S0 (%)")
    ax.legend(ncol=2, loc="upper left")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=year_norm, cmap=SEQUENTIAL), ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("GCAM node")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    matrix = province.groupby(["region", "year"])["s2_increment_share_total_benefit_pct"].median().unstack("year")
    matrix = matrix.reindex(columns=YEARS[1:]).sort_values(2060, ascending=False)
    draw_heatmap(
        fig, ax, matrix, DIVERGING, "S1-S2 share of total ENS benefit (%)",
        norm=TwoSlopeNorm(vmin=0, vcenter=50, vmax=100), annotate=False,
    )
    ax.set_xlabel("GCAM node")
    selected_rows = list(range(0, len(matrix), 5))
    if selected_rows[-1] != len(matrix) - 1:
        selected_rows.append(len(matrix) - 1)
    ax.set_yticks(selected_rows)
    ax.set_yticklabels([matrix.index[index] for index in selected_rows], fontsize=5.0)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    top = endpoint.nlargest(12, "residual_s1_gwh").sort_values("residual_s1_gwh")
    y = np.arange(len(top))
    ax.hlines(y, top["residual_s2_gwh"], top["residual_s1_gwh"], color=LIGHT, lw=1.2, zorder=1)
    ax.scatter(top["residual_s1_gwh"], y, color=GOLD, s=22, edgecolor="white", linewidth=0.35, label="S1", zorder=2)
    ax.scatter(top["residual_s2_gwh"], y, color=GREEN, s=22, edgecolor="white", linewidth=0.35, label="S2", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, top["region"])
    ax.set_xlabel("Residual worst-event ENS, 168 h (GWh)")
    ax.legend(ncol=2, loc="lower right")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    corr = correlations.groupby(["storage_pathway", "predictor", "year"])["spearman_rho"].median().unstack("year")
    rows = []
    for predictor in PREDICTORS:
        for pathway in ["S1_ORDERLY", "S2_ACCELERATED"]:
            row = corr.loc[(pathway, predictor)]
            row.name = f"{PREDICTOR_LABEL[predictor]} {PATH_LABEL[pathway]}"
            rows.append(row)
    matrix = pd.DataFrame(rows).reindex(columns=YEARS)
    draw_heatmap(
        fig, ax, matrix, CORRMAP, "Median Spearman rho",
        norm=TwoSlopeNorm(vmin=-0.65, vcenter=0, vmax=0.65), fmt=".2f", annotate=False,
    )
    ax.set_xlabel("GCAM node")
    ax.tick_params(axis="y", labelsize=3.9)
    panel(ax, "d")
    save_figure(fig, "figure_6_three_path_capacity_operation_translation")


def figure_7(duration_stage: pd.DataFrame, event_stage: pd.DataFrame, floor: pd.DataFrame, matched: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.50)

    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for year in KEY_YEARS:
        for transition in ["01", "12"]:
            row = duration_stage[duration_stage["year"].eq(year)].groupby("window_hours")[f"operation_reduction_{transition}_pct_s0"].median()
            row.name = f"{year} {TRANSITION_LABEL[transition]}"
            rows.append(row)
    matrix = pd.DataFrame(rows).reindex(columns=WINDOWS)
    matrix.columns = [f"{value} h" for value in matrix.columns]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "ENS reduction vs S0 (%)", fmt=".1f")
    ax.set_xlabel("Event duration")
    ax.tick_params(axis="y", labelsize=4.8)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    endpoint = event_stage[event_stage["year"].eq(2060)].groupby(
        ["event_type", "window_hours"], as_index=False
    )[["S0_BASE", "operation_reduction_01", "operation_reduction_12"]].median()
    endpoint["reduction_01_pct"] = safe_ratio(endpoint["operation_reduction_01"], endpoint["S0_BASE"], 100.0)
    endpoint["reduction_12_pct"] = safe_ratio(endpoint["operation_reduction_12"], endpoint["S0_BASE"], 100.0)
    columns = []
    values = {}
    for transition in ["01", "12"]:
        for window in WINDOWS:
            label = f"{TRANSITION_LABEL[transition]}\n{window} h"
            columns.append(label)
            series = endpoint[endpoint["window_hours"].eq(window)].set_index("event_type")[f"reduction_{transition}_pct"]
            values[label] = series
    matrix = pd.DataFrame(values).reindex(EVENT_ORDER)
    matrix.index = [EVENT_LABEL[value] for value in matrix.index]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "ENS reduction vs S0 (%)", fmt=".0f")
    ax.tick_params(axis="x", labelsize=4.4)
    ax.tick_params(axis="y", labelsize=4.5)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    endpoint_floor = floor[floor["year"].eq(2060)].groupby(
        ["storage_pathway", "event_type", "window_hours"]
    )["soc_floor_share_pct"].median()
    rows = []
    for event_type in EVENT_ORDER:
        for pathway in ["S1_ORDERLY", "S2_ACCELERATED"]:
            row = endpoint_floor.loc[(pathway, event_type)]
            row.name = f"{EVENT_LABEL[event_type]} {PATH_LABEL[pathway]}"
            rows.append(row)
    matrix = pd.DataFrame(rows).reindex(columns=WINDOWS)
    matrix.columns = [f"{value} h" for value in matrix.columns]
    draw_heatmap(
        fig, ax, matrix, SEQUENTIAL, "Provinces reaching SOC floor (%)",
        norm=Normalize(0, 100), fmt=".0f",
    )
    ax.set_xlabel("Event duration")
    ax.tick_params(axis="y", labelsize=3.9)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    visible = matched[matched["direct_storage_eens_reduction_gwh"].gt(1e-6)].copy()
    normal = visible[~visible["negative_total_effect"]]
    negative = visible[visible["negative_total_effect"]]
    ax.scatter(normal["direct_storage_eens_reduction_gwh"], normal["portfolio_substitution_gwh"], s=5, color=GRAY, alpha=0.12, linewidth=0)
    ax.scatter(
        negative["direct_storage_eens_reduction_gwh"], negative["portfolio_substitution_gwh"],
        s=14, facecolor=CORAL, edgecolor="white", linewidth=0.3, alpha=0.88,
        label=f"Total effect < 0 (n={len(negative)})",
    )
    xmax = max(1.0, float(matched["direct_storage_eens_reduction_gwh"].max()))
    xline = np.geomspace(0.01, xmax, 200)
    ax.plot(xline, -xline, color=INK, lw=0.65, ls="--")
    ax.axhline(0, color=LIGHT, lw=0.65)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("Direct storage effect (GWh)")
    ax.set_ylabel("Portfolio-substitution effect (GWh)")
    ax.legend(loc="lower left")
    panel(ax, "d")
    save_figure(fig, "figure_7_three_path_event_duration_boundaries")


def write_outputs(
    stage: pd.DataFrame,
    trajectory: pd.DataFrame,
    power: pd.DataFrame,
    robustness: pd.DataFrame,
    province: pd.DataFrame,
    correlations: pd.DataFrame,
    endpoint: pd.DataFrame,
    event_stage: pd.DataFrame,
    duration_stage: pd.DataFrame,
    floor: pd.DataFrame,
    matched: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    stage.to_csv(TABLE / "national_three_path_stage_cases.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(TABLE / "national_three_path_trajectory.csv", index=False, encoding="utf-8-sig")
    power.to_csv(TABLE / "storage_power_pathways.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(TABLE / "ambition_effect_robustness_by_storage.csv", index=False, encoding="utf-8-sig")
    province.to_csv(TABLE / "province_three_path_stage_effects.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(TABLE / "province_correlations_s1_s2.csv", index=False, encoding="utf-8-sig")
    endpoint.to_csv(TABLE / "province_residual_risk_s1_s2_2060.csv", index=False, encoding="utf-8-sig")
    event_stage.to_csv(TABLE / "event_three_path_stage_effects.csv", index=False, encoding="utf-8-sig")
    duration_stage.to_csv(TABLE / "duration_three_path_stage_effects.csv", index=False, encoding="utf-8-sig")
    floor.to_csv(TABLE / "event_soc_floor_s1_s2.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(TABLE / "direct_portfolio_total_effects.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(TABLE / "climate_quantile_three_path_sensitivity_2060.csv", index=False, encoding="utf-8-sig")

    key = trajectory[trajectory["year"].isin([2030, 2040, 2050, 2060])].copy()
    summary = {
        "rows": {
            "national_stage_cases": int(len(stage)),
            "province_stage_cases": int(len(province)),
            "event_stage_cases": int(len(event_stage)),
            "matched_cases": int(len(matched)),
        },
        "key_years": {},
        "negative_total_effect_cases": int(matched["negative_total_effect"].sum()),
    }
    for row in key.itertuples(index=False):
        year = int(row.year)
        summary["key_years"][str(year)] = {
            "capacity_reduction_01_gw": float(getattr(row, "capacity_reduction_01_median")),
            "capacity_reduction_12_gw": float(getattr(row, "capacity_reduction_12_median")),
            "capacity_reduction_02_gw": float(getattr(row, "capacity_reduction_02_median")),
            "operation_reduction_01_gwh": float(getattr(row, "operation_reduction_01_median")),
            "operation_reduction_12_gwh": float(getattr(row, "operation_reduction_12_median")),
            "operation_reduction_02_gwh": float(getattr(row, "operation_reduction_02_median")),
            "conversion_01": float(getattr(row, "conversion_01_median")),
            "conversion_12": float(getattr(row, "conversion_12_median")),
            "conversion_02": float(getattr(row, "conversion_02_median")),
        }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    apply_style()
    stage, trajectory, power = build_national_stage_data()
    robustness = build_ambition_robustness()
    province, correlations, endpoint = build_province_stage_data()
    event_stage, floor, matched = build_event_stage_data()
    duration_stage = build_duration_stage_data()
    sensitivity = build_climate_stage_sensitivity()
    figure_4(stage, trajectory)
    figure_6(stage, province, correlations, endpoint)
    figure_7(duration_stage, event_stage, floor, matched)
    write_outputs(
        stage, trajectory, power, robustness, province, correlations, endpoint,
        event_stage, duration_stage, floor, matched, sensitivity,
    )
    print(json.dumps(json.load((OUT / "summary.json").open(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
