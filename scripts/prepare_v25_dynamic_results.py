from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / "outputs" / "v19"
V24 = ROOT / "outputs" / "v24"
OUT = ROOT / "outputs" / "v25_dynamic_results"
FIG = OUT / "figures"
TABLE = OUT / "tables"

LONG_FILE = V19 / "long_term_chain" / "national_long_term_summary.csv"
ADEQUACY_FILE = V19 / "adequacy" / "province_capacity_adequacy_with_mechanisms.csv"
EVENT_FILE = V24 / "dynamic_event_dispatch" / "province_event_duration_reliability.csv"
DURATION_FILE = V24 / "dynamic_event_dispatch" / "province_duration_reliability.csv"
STORAGE_FILE = V24 / "dynamic_climate_uncertainty" / "storage_effect_uncertainty.csv"
DIRECT_FILE = V24 / "dynamic_matched_storage_dispatch" / "matched_storage_direct_effects.csv"

INK = "#25272A"
GRAY = "#858B91"
LIGHT = "#D9DDE1"
PALE = "#F4F5F4"
TEAL = "#4FAFA6"
CORAL = "#E9785D"
BLUE = "#4B91C3"
GOLD = "#E5BC56"
PURPLE = "#8176A7"
GREEN = "#54A878"

AMBITION_ORDER = ["CN_NDC7_NZ2060", "CN_NDC10_NZ2060", "IPCC15_ACCELERATED_2050"]
AMBITION_LABEL = {
    "CN_NDC7_NZ2060": "NDC-7",
    "CN_NDC10_NZ2060": "NDC-10",
    "IPCC15_ACCELERATED_2050": "IPCC-1.5C",
}
AMBITION_COLOR = {
    "CN_NDC7_NZ2060": GRAY,
    "CN_NDC10_NZ2060": GOLD,
    "IPCC15_ACCELERATED_2050": PURPLE,
}
AMBITION_MARKER = {
    "CN_NDC7_NZ2060": "o",
    "CN_NDC10_NZ2060": "s",
    "IPCC15_ACCELERATED_2050": "D",
}
EVENT_ORDER = [
    "compound_heat_low_vre",
    "heat_load",
    "low_vre",
    "cold_derating",
    "transfer_restriction",
]
EVENT_LABEL = {
    "compound_heat_low_vre": "Compound heat-low VRE",
    "heat_load": "Heat load",
    "low_vre": "Low wind and solar",
    "cold_derating": "Cold derating",
    "transfer_restriction": "Transfer restriction",
}
EVENT_COLOR = {
    "compound_heat_low_vre": PURPLE,
    "heat_load": CORAL,
    "low_vre": TEAL,
    "cold_derating": BLUE,
    "transfer_restriction": GOLD,
}
YEARS = [2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
KEY_YEARS = [2030, 2050, 2060]
PREDICTORS = [
    "vre_anchored_share",
    "weather_derating_exposure_share",
    "firm_capacity_retention",
    "storage_power_to_peak",
    "storage_energy_to_peak_hours",
    "transfer_dependence",
    "delta_peak",
]
PREDICTOR_LABEL = {
    "vre_anchored_share": "VRE capacity share",
    "weather_derating_exposure_share": "Weather derating exposure",
    "firm_capacity_retention": "Firm-capacity retention",
    "storage_power_to_peak": "Storage power / peak load",
    "storage_energy_to_peak_hours": "Storage energy / peak load",
    "transfer_dependence": "Transfer dependence",
    "delta_peak": "Peak-load shock",
}

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "v25_seq", ["#F4F3EF", "#C8DBD7", "#74B5AE", "#E4C56B", "#D77962", "#85394C"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "v25_div", ["#B84A55", "#E7A39A", "#F7F7F4", "#9BCBC5", "#357F78"]
)
CORRMAP = LinearSegmentedColormap.from_list(
    "v25_corr", ["#365F7D", "#A9C7D1", "#F7F7F4", "#EBA493", "#A43B4A"]
)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.labelsize": 6.2,
        "axes.titlesize": 6.2,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "legend.fontsize": 5.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.1,
        "ytick.major.size": 2.1,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    })


def panel(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def draw_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: pd.DataFrame,
    cmap,
    colorbar_label: str,
    norm=None,
    fmt: str = ".1f",
    annotate: bool = True,
) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if norm is None:
        lo, hi = np.quantile(finite, [0.02, 0.98])
        if np.isclose(lo, hi):
            hi = lo + 1e-9
        norm = Normalize(float(lo), float(hi))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not np.isfinite(value):
                    text = "-"
                    color = GRAY
                else:
                    text = format(value, fmt)
                    rel = float(np.clip(norm(value), 0, 1))
                    color = "white" if rel < 0.12 or rel > 0.82 else INK
                ax.text(j, i, text, ha="center", va="center", fontsize=4.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=4.8, width=0.4, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_long_term() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_csv(LONG_FILE)
    frame = frame[frame["risk_level"].eq("R50") & frame["storage_pathway"].eq("S1_ORDERLY")].copy()
    denominator = frame[[
        "thermal_generation_twh", "hydro_generation_twh", "wind_generation_twh", "solar_generation_twh"
    ]].sum(axis=1)
    frame["vre_generation_share_pct"] = 100 * (
        frame["wind_generation_twh"] + frame["solar_generation_twh"]
    ) / denominator.clip(lower=1e-9)
    frame["weather_exposure_pct"] = 100 * frame["weather_derating_exposure_share"]
    frame["firm_retention_pct"] = 100 * frame["firm_capacity_retention"]
    metrics = [
        "power_sector_co2_mtc",
        "vre_generation_share_pct",
        "weather_exposure_pct",
        "firm_retention_pct",
        "positive_gap_exposure_gw",
        "storage_power_gw",
        "storage_energy_gwh",
    ]
    summary = frame.groupby(["ambition_pathway", "year"], as_index=False)[metrics].agg(["median", "min", "max"])
    summary.columns = ["_".join([str(v) for v in col if v]).rstrip("_") for col in summary.columns]
    summary = summary.rename(columns={"ambition_pathway_": "ambition_pathway", "year_": "year"})
    return frame, summary


def build_storage_trajectory() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_csv(STORAGE_FILE)
    frame = frame[frame["ensemble_quantile"].eq("median") & frame["window_hours"].eq(168)].copy()
    frame["conversion_ratio"] = np.where(
        frame["capacity_gap_reduction_pct"].abs().gt(1e-9),
        frame["eens_reduction_pct"] / frame["capacity_gap_reduction_pct"],
        np.nan,
    )
    metrics = [
        "capacity_gap_s0_gw", "capacity_gap_s2_gw", "capacity_gap_reduction_gw",
        "capacity_gap_reduction_pct", "eens_s0_gwh", "eens_s2_gwh",
        "eens_reduction_gwh", "eens_reduction_pct", "conversion_ratio",
    ]
    summary = frame.groupby("year", as_index=False)[metrics].agg(["median", "min", "max"])
    summary.columns = ["_".join([str(v) for v in col if v]).rstrip("_") for col in summary.columns]
    summary = summary.rename(columns={"year_": "year"})
    return frame, summary


def build_province_storage() -> pd.DataFrame:
    duration = read_csv(DURATION_FILE)
    duration = duration[duration["risk_level"].eq("R50") & duration["window_hours"].eq(168)].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    pivot = duration.pivot_table(index=keys, columns="storage_pathway", values="worst_eens_gwh").reset_index()
    pivot["storage_benefit_gwh"] = pivot["S0_BASE"] - pivot["S2_ACCELERATED"]
    pivot["storage_benefit_pct"] = np.where(
        pivot["S0_BASE"].gt(1e-6), 100 * pivot["storage_benefit_gwh"] / pivot["S0_BASE"], 0.0
    )
    return pivot


def build_dynamic_correlations(province_storage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adequacy = read_csv(ADEQUACY_FILE)
    adequacy = adequacy[
        adequacy["risk_level"].eq("R50") & adequacy["storage_pathway"].eq("S2_ACCELERATED")
    ].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year"]
    merged = province_storage.merge(
        adequacy[keys + PREDICTORS + ["adequacy_gap_gw"]], on=keys, validate="one_to_one"
    )
    rows = []
    for (climate, ambition, year), group in merged.groupby(["climate_scenario", "ambition_pathway", "year"]):
        for predictor in PREDICTORS:
            rows.append({
                "climate_scenario": climate,
                "ambition_pathway": ambition,
                "year": int(year),
                "predictor": predictor,
                "spearman_rho": float(group[[predictor, "S2_ACCELERATED"]].corr(method="spearman").iloc[0, 1]),
            })
    correlations = pd.DataFrame(rows)
    return merged, correlations


def build_duration_event_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    storage = read_csv(STORAGE_FILE)
    duration = storage[storage["ensemble_quantile"].eq("median") & storage["year"].isin(KEY_YEARS)].copy()
    duration_summary = duration.groupby(["year", "window_hours"], as_index=False)[
        ["capacity_gap_reduction_pct", "eens_reduction_pct"]
    ].median()
    duration_summary["conversion_ratio"] = np.where(
        duration_summary["capacity_gap_reduction_pct"].abs().gt(1e-9),
        duration_summary["eens_reduction_pct"] / duration_summary["capacity_gap_reduction_pct"],
        np.nan,
    )

    event = read_csv(EVENT_FILE)
    event = event[event["risk_level"].eq("R50")].copy()
    group_keys = ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type", "storage_pathway"]
    national = event.groupby(group_keys, as_index=False)["eens_gwh"].sum()
    pivot_keys = ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type"]
    pivot = national.pivot_table(index=pivot_keys, columns="storage_pathway", values="eens_gwh").reset_index()
    pivot["eens_reduction_gwh"] = pivot["S0_BASE"] - pivot["S2_ACCELERATED"]
    pivot["eens_reduction_pct"] = np.where(
        pivot["S0_BASE"].gt(1e-6), 100 * pivot["eens_reduction_gwh"] / pivot["S0_BASE"], np.nan
    )
    event_summary = pivot.groupby(["year", "window_hours", "event_type"], as_index=False)[
        ["S0_BASE", "S2_ACCELERATED", "eens_reduction_gwh", "eens_reduction_pct"]
    ].median()

    floor = event[event["storage_pathway"].eq("S2_ACCELERATED")].copy()
    floor["reaches_floor"] = floor["hours_at_soc_floor"].gt(0).astype(int)
    floor_scenario = floor.groupby(
        ["climate_scenario", "ambition_pathway", "year", "window_hours", "event_type"], as_index=False
    )["reaches_floor"].sum()
    floor_scenario["soc_floor_share_pct"] = 100 * floor_scenario["reaches_floor"] / 31
    floor_summary = floor_scenario.groupby(["year", "window_hours", "event_type"], as_index=False)[
        ["reaches_floor", "soc_floor_share_pct"]
    ].median()
    return duration_summary, event_summary, floor_summary


def build_matched_effects() -> pd.DataFrame:
    event = read_csv(EVENT_FILE)
    event = event[event["risk_level"].eq("R50")].copy()
    keys = ["climate_scenario", "ambition_pathway", "region", "year", "window_hours", "event_type"]
    total = event.pivot_table(index=keys, columns="storage_pathway", values="eens_gwh").reset_index()
    total["total_effect_gwh"] = total["S0_BASE"] - total["S2_ACCELERATED"]
    direct = read_csv(DIRECT_FILE)
    out = total.merge(direct[keys + ["direct_storage_eens_reduction_gwh"]], on=keys, validate="one_to_one")
    out["portfolio_substitution_gwh"] = out["total_effect_gwh"] - out["direct_storage_eens_reduction_gwh"]
    out["negative_total_effect"] = out["total_effect_gwh"].lt(-1e-8)
    return out


def figure_1(long_frame: pd.DataFrame, long_summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    for ambition in AMBITION_ORDER:
        group = long_summary[long_summary["ambition_pathway"].eq(ambition)].sort_values("year")
        x = group["year"].to_numpy()
        y = group["power_sector_co2_mtc_median"].to_numpy()
        ax.fill_between(x, group["power_sector_co2_mtc_min"], group["power_sector_co2_mtc_max"], color=AMBITION_COLOR[ambition], alpha=0.13, linewidth=0)
        ax.plot(x, y, color=AMBITION_COLOR[ambition], lw=1.4, marker=AMBITION_MARKER[ambition], ms=3.1, label=AMBITION_LABEL[ambition])
    ax.set_ylabel("Power-sector CO2 (MtC)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    ax.legend(ncol=3, loc="upper right", columnspacing=0.8, handlelength=1.4)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    matrix = long_summary.pivot(index="ambition_pathway", columns="year", values="vre_generation_share_pct_median").reindex(AMBITION_ORDER)
    matrix.index = [AMBITION_LABEL[v] for v in matrix.index]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "Wind and solar generation (%)", fmt=".0f")
    ax.set_xlabel("GCAM node")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    matrix = long_summary.pivot(index="ambition_pathway", columns="year", values="weather_exposure_pct_median").reindex(AMBITION_ORDER)
    matrix.index = [AMBITION_LABEL[v] for v in matrix.index]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "Weather-derating exposure (%)", fmt=".0f")
    ax.set_xlabel("GCAM node")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    for ambition in AMBITION_ORDER:
        group = long_summary[long_summary["ambition_pathway"].eq(ambition)].sort_values("year")
        x = group["year"].to_numpy()
        y = group["positive_gap_exposure_gw_median"].to_numpy()
        ax.fill_between(x, group["positive_gap_exposure_gw_min"], group["positive_gap_exposure_gw_max"], color=AMBITION_COLOR[ambition], alpha=0.13, linewidth=0)
        ax.plot(x, y, color=AMBITION_COLOR[ambition], lw=1.4, marker=AMBITION_MARKER[ambition], ms=3.1)
    ax.set_ylabel("Positive capacity-gap exposure (GW)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    panel(ax, "d")
    save(fig, "figure_1_dynamic_ambition_risk_chain")


def figure_2(storage_frame: pd.DataFrame, trajectory: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    for pathway, label, color, ls in [
        ("capacity_gap_s0_gw", "S0", GRAY, "--"),
        ("capacity_gap_s2_gw", "S2", GREEN, "-"),
    ]:
        ax.plot(trajectory["year"], trajectory[f"{pathway}_median"], color=color, lw=1.5, marker="o", ms=3.0, ls=ls, label=label)
        ax.fill_between(trajectory["year"], trajectory[f"{pathway}_min"], trajectory[f"{pathway}_max"], color=color, alpha=0.12, linewidth=0)
    ax.set_ylabel("Positive capacity gap (GW)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    ax.legend(ncol=2, loc="upper left")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for pathway, label, color, ls in [
        ("eens_s0_gwh", "S0", GRAY, "--"),
        ("eens_s2_gwh", "S2", GREEN, "-"),
    ]:
        ax.plot(trajectory["year"], trajectory[f"{pathway}_median"] / 1000, color=color, lw=1.5, marker="o", ms=3.0, ls=ls, label=label)
        ax.fill_between(trajectory["year"], trajectory[f"{pathway}_min"] / 1000, trajectory[f"{pathway}_max"] / 1000, color=color, alpha=0.12, linewidth=0)
    ax.set_ylabel("Worst-event EENS, 168 h (TWh)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(trajectory["year"], trajectory["capacity_gap_reduction_pct_median"], color=BLUE, marker="s", ms=3.2, lw=1.5, label="Capacity")
    ax.plot(trajectory["year"], trajectory["eens_reduction_pct_median"], color=CORAL, marker="o", ms=3.2, lw=1.5, label="Operation")
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_ylabel("S0-to-S2 risk reduction (%)")
    ax.set_xlabel("GCAM node")
    ax.set_xticks(YEARS[::2], YEARS[::2])
    ax.legend(ncol=2, loc="upper left")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    matrix = storage_frame.groupby(["ambition_pathway", "year"])["conversion_ratio"].median().unstack("year").reindex(AMBITION_ORDER)
    matrix.index = [AMBITION_LABEL[v] for v in matrix.index]
    finite = matrix.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "Capacity-to-operation conversion", norm=Normalize(0, max(2.2, float(np.nanmax(finite)))), fmt=".2f")
    ax.set_xlabel("GCAM node")
    panel(ax, "d")
    save(fig, "figure_2_dynamic_storage_mediation")


def figure_3(storage_frame: pd.DataFrame, province_storage: pd.DataFrame, merged: pd.DataFrame, correlations: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18], hspace=0.50, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    plot = storage_frame[storage_frame["year"].gt(2025)].copy()
    year_norm = Normalize(2030, 2060)
    for ambition in AMBITION_ORDER:
        group = plot[plot["ambition_pathway"].eq(ambition)]
        ax.scatter(group["capacity_gap_reduction_pct"], group["eens_reduction_pct"], c=group["year"], cmap=SEQUENTIAL, norm=year_norm, marker=AMBITION_MARKER[ambition], s=24, alpha=0.78, edgecolor="white", linewidth=0.35, label=AMBITION_LABEL[ambition])
    high = max(plot["capacity_gap_reduction_pct"].max(), plot["eens_reduction_pct"].max()) * 1.05
    ax.plot([0, high], [0, high], color=LIGHT, lw=0.8, zorder=0)
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)
    ax.set_xlabel("Capacity-gap reduction (%)")
    ax.set_ylabel("168 h EENS reduction (%)")
    ax.legend(ncol=1, loc="upper left")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=year_norm, cmap=SEQUENTIAL), ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("GCAM node")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    province_matrix = province_storage.groupby(["region", "year"])["storage_benefit_pct"].median().unstack("year").reindex(columns=YEARS)
    province_matrix = province_matrix.sort_values(2060, ascending=False)
    values = province_matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lo = min(-2.0, float(np.quantile(finite, 0.01)))
    hi = max(8.0, float(np.quantile(finite, 0.97)))
    draw_heatmap(fig, ax, province_matrix, DIVERGING, "Province-level EENS reduction (%)", norm=TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi), annotate=False)
    ax.set_xlabel("GCAM node")
    selected_rows = list(range(0, len(province_matrix), 5))
    if selected_rows[-1] != len(province_matrix) - 1:
        selected_rows.append(len(province_matrix) - 1)
    ax.set_yticks(selected_rows)
    ax.set_yticklabels([province_matrix.index[i] for i in selected_rows], fontsize=5.0)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    endpoint = merged[merged["year"].eq(2060)].groupby("region", as_index=False).agg(
        residual_eens_gwh=("S2_ACCELERATED", "median"),
        storage_hours=("storage_energy_to_peak_hours", "median"),
        vre_share=("vre_anchored_share", "median"),
    )
    ax.scatter(endpoint["storage_hours"], endpoint["residual_eens_gwh"], color=TEAL, s=26, alpha=0.76, edgecolor="white", linewidth=0.35)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("Storage energy / peak load (h)")
    ax.set_ylabel("Residual worst-event EENS, 168 h (GWh)")
    for row in endpoint.nlargest(6, "residual_eens_gwh").itertuples(index=False):
        ax.annotate(row.region, (row.storage_hours, row.residual_eens_gwh), xytext=(3, 2), textcoords="offset points", fontsize=4.6, color=INK)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    matrix = correlations.groupby(["predictor", "year"])["spearman_rho"].median().unstack("year").reindex(PREDICTORS)
    matrix.index = [PREDICTOR_LABEL[v] for v in matrix.index]
    draw_heatmap(fig, ax, matrix, CORRMAP, "Median Spearman rho", norm=TwoSlopeNorm(vmin=-0.65, vcenter=0, vmax=0.65), fmt=".2f")
    ax.set_xlabel("GCAM node")
    ax.tick_params(axis="y", labelsize=4.5)
    panel(ax, "d")
    save(fig, "figure_3_capacity_operation_translation")


def figure_4(duration: pd.DataFrame, events: pd.DataFrame, floor: pd.DataFrame, matched: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.25, 4.95))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    matrix = duration.pivot(index="year", columns="window_hours", values="eens_reduction_pct").reindex(index=KEY_YEARS, columns=[72, 168, 336])
    matrix.index = [str(v) for v in matrix.index]
    matrix.columns = [f"{v} h" for v in matrix.columns]
    draw_heatmap(fig, ax, matrix, SEQUENTIAL, "EENS reduction (%)", fmt=".1f")
    ax.set_xlabel("Event duration")
    ax.set_ylabel("GCAM node")
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    endpoint = events[events["year"].eq(2060)].pivot(index="event_type", columns="window_hours", values="eens_reduction_pct").reindex(index=EVENT_ORDER, columns=[72, 168, 336])
    endpoint.index = [EVENT_LABEL[v] for v in endpoint.index]
    endpoint.columns = [f"{v} h" for v in endpoint.columns]
    draw_heatmap(fig, ax, endpoint, SEQUENTIAL, "EENS reduction (%)", fmt=".0f")
    ax.set_xlabel("Event duration")
    ax.tick_params(axis="y", labelsize=4.7)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    endpoint = floor[floor["year"].eq(2060)].pivot(index="event_type", columns="window_hours", values="soc_floor_share_pct").reindex(index=EVENT_ORDER, columns=[72, 168, 336])
    endpoint.index = [EVENT_LABEL[v] for v in endpoint.index]
    endpoint.columns = [f"{v} h" for v in endpoint.columns]
    draw_heatmap(fig, ax, endpoint, SEQUENTIAL, "Provinces reaching SOC floor (%)", norm=Normalize(0, 100), fmt=".0f")
    ax.set_xlabel("Event duration")
    ax.tick_params(axis="y", labelsize=4.7)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    visible = matched[matched["direct_storage_eens_reduction_gwh"].gt(1e-6)].copy()
    normal = visible[~visible["negative_total_effect"]]
    negative = visible[visible["negative_total_effect"]]
    ax.scatter(normal["direct_storage_eens_reduction_gwh"], normal["portfolio_substitution_gwh"], s=5, color=GRAY, alpha=0.12, linewidth=0)
    ax.scatter(negative["direct_storage_eens_reduction_gwh"], negative["portfolio_substitution_gwh"], s=14, facecolor=CORAL, edgecolor="white", linewidth=0.3, alpha=0.88, label=f"Total effect < 0 (n={len(negative)})")
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
    save(fig, "figure_4_event_duration_failure_boundaries")


def write_outputs(
    long_summary: pd.DataFrame,
    trajectory: pd.DataFrame,
    province_storage: pd.DataFrame,
    merged: pd.DataFrame,
    correlations: pd.DataFrame,
    duration: pd.DataFrame,
    events: pd.DataFrame,
    floor: pd.DataFrame,
    matched: pd.DataFrame,
) -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    long_summary.to_csv(TABLE / "table_1_dynamic_ambition_pathways.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(TABLE / "table_2_dynamic_storage_trajectory.csv", index=False, encoding="utf-8-sig")
    duration.to_csv(TABLE / "table_3_duration_year_sensitivity.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TABLE / "table_4_event_specific_effects.csv", index=False, encoding="utf-8-sig")
    floor.to_csv(TABLE / "table_5_soc_floor_exposure.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(TABLE / "table_6_dynamic_province_correlations.csv", index=False, encoding="utf-8-sig")
    province_storage.to_csv(TABLE / "table_7_province_dynamic_storage_effects.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(TABLE / "table_8_direct_and_portfolio_effects.csv", index=False, encoding="utf-8-sig")

    province_median = province_storage.groupby(["region", "year"], as_index=False).agg(
        storage_benefit_gwh=("storage_benefit_gwh", "median"),
        storage_benefit_pct=("storage_benefit_pct", "median"),
        residual_eens_gwh=("S2_ACCELERATED", "median"),
    )
    province_summary = province_median.groupby("year", as_index=False).agg(
        median_benefit_pct=("storage_benefit_pct", "median"),
        q25_benefit_pct=("storage_benefit_pct", lambda value: value.quantile(0.25)),
        q75_benefit_pct=("storage_benefit_pct", lambda value: value.quantile(0.75)),
        minimum_benefit_pct=("storage_benefit_pct", "min"),
        maximum_benefit_pct=("storage_benefit_pct", "max"),
        positive_provinces=("storage_benefit_pct", lambda value: int((value > 1e-8).sum())),
        negative_provinces=("storage_benefit_pct", lambda value: int((value < -1e-8).sum())),
    )
    province_summary.to_csv(TABLE / "table_9_province_dynamic_summary.csv", index=False, encoding="utf-8-sig")

    correlation_summary = correlations.groupby(["predictor", "year"], as_index=False).agg(
        median_spearman_rho=("spearman_rho", "median"),
        minimum_spearman_rho=("spearman_rho", "min"),
        maximum_spearman_rho=("spearman_rho", "max"),
    )
    correlation_summary.to_csv(TABLE / "table_10_correlation_medians.csv", index=False, encoding="utf-8-sig")

    endpoint = merged[merged["year"].eq(2060)].groupby("region", as_index=False).agg(
        residual_eens_gwh=("S2_ACCELERATED", "median"),
        storage_energy_to_peak_hours=("storage_energy_to_peak_hours", "median"),
        storage_power_to_peak=("storage_power_to_peak", "median"),
        vre_anchored_share=("vre_anchored_share", "median"),
        weather_derating_exposure_share=("weather_derating_exposure_share", "median"),
        firm_capacity_retention=("firm_capacity_retention", "median"),
    ).sort_values("residual_eens_gwh", ascending=False)
    endpoint.to_csv(TABLE / "table_11_province_residual_risk_2060.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "rows": {
            "long_term": int(len(read_csv(LONG_FILE))),
            "event_dispatch": int(len(read_csv(EVENT_FILE))),
            "matched_cases": int(len(matched)),
            "province_storage": int(len(province_storage)),
        },
        "negative_total_effect_cases": int(matched["negative_total_effect"].sum()),
        "negative_direct_effect_cases": int((matched["direct_storage_eens_reduction_gwh"] < -1e-8).sum()),
        "negative_case_provinces": sorted(matched.loc[matched["negative_total_effect"], "region"].unique().tolist()),
        "negative_case_years": sorted(matched.loc[matched["negative_total_effect"], "year"].unique().astype(int).tolist()),
        "worst_total_effect_gwh": float(matched["total_effect_gwh"].min()),
    }
    (OUT / "metrics_v25.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def audit_outputs() -> None:
    expected = [
        FIG / "figure_1_dynamic_ambition_risk_chain.png",
        FIG / "figure_2_dynamic_storage_mediation.png",
        FIG / "figure_3_capacity_operation_translation.png",
        FIG / "figure_4_event_duration_failure_boundaries.png",
    ]
    missing = [str(path) for path in expected if not path.exists() or path.stat().st_size < 50_000]
    if missing:
        raise RuntimeError(f"Missing or undersized figures: {missing}")
    trajectory = read_csv(TABLE / "table_2_dynamic_storage_trajectory.csv")
    if trajectory["year"].tolist() != YEARS:
        raise RuntimeError("Dynamic trajectory does not contain all eight GCAM nodes")
    matched = read_csv(TABLE / "table_8_direct_and_portfolio_effects.csv")
    if (matched["direct_storage_eens_reduction_gwh"] < -1e-8).any():
        raise RuntimeError("Negative direct storage effect found")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    long_frame, long_summary = build_long_term()
    storage_frame, trajectory = build_storage_trajectory()
    province_storage = build_province_storage()
    merged, correlations = build_dynamic_correlations(province_storage)
    duration, events, floor = build_duration_event_tables()
    matched = build_matched_effects()
    figure_1(long_frame, long_summary)
    figure_2(storage_frame, trajectory)
    figure_3(storage_frame, province_storage, merged, correlations)
    figure_4(duration, events, floor, matched)
    write_outputs(long_summary, trajectory, province_storage, merged, correlations, duration, events, floor, matched)
    audit_outputs()
    print(OUT)


if __name__ == "__main__":
    main()
