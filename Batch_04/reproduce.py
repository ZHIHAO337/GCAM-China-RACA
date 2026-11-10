"""Portable reproduction from archived long-term outputs, not a native GCAM run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
import build_formal_climate_inputs as climate
import analyze_v19_long_term_chain as chain
import run_v20_climate_uncertainty_sensitivity as uncertainty
import reviewer_correct_layers_20261104 as corrected
from event_supply import event_transfer_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-events', action='store_true', help='Replay all 117180 event and 13020 matched cases')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = ROOT/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    assumptions = json.loads((ROOT/'config/model_assumptions.json').read_text(encoding='utf-8'))
    master = pd.read_csv(ROOT/'data/processed/province_master.csv')
    hazards = pd.read_csv(ROOT/'data/formal/cmip6_node_hazard_ensemble.csv')
    historical = pd.read_csv(ROOT/'data/formal/era5_historical_hazard_multipliers.csv')
    risk = pd.concat([climate.build_risk_parameters(master, hazards, historical, assumptions, s, q).assign(ensemble_quantile=q)
                      for s in climate.SCENARIOS for q in ['p10','median','p90']], ignore_index=True)
    risk_ref = pd.read_csv(ROOT/'data/formal/risk_parameter_uncertainty.csv')
    rk = ['province','climate_scenario','risk_level','year','ensemble_quantile']
    fields = ['delta_peak','derate_pv','derate_wind','derate_hydro','derate_thermal','trade_loss','climate_amplification','equivalent_reserve_margin']
    joined = risk.merge(risk_ref, on=rk, suffixes=('_new','_ref'), validate='one_to_one')
    risk_error = max(float((joined[k+'_new']-joined[k+'_ref']).abs().max()) for k in fields)
    assert len(joined) == 5952 and risk_error < 1e-10
    for s in climate.SCENARIOS:
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/analyze_v4_integrated_adequacy.py'),
                        str(ROOT/f'outputs/v19/{s}/processed/province_scenario_metrics.csv'),
                        '--climate', s, '--output-dir', str(ROOT/f'outputs/v19/{s}/adequacy')], check=True)
    chain.main()
    base = pd.read_csv(ROOT/'outputs/v19/adequacy/province_capacity_adequacy_with_mechanisms.csv')
    expanded = uncertainty.build_uncertain_adequacy(base, risk)
    ref = pd.read_parquet(ROOT/'reference/capacity.parquet')
    keys = corrected.KEYS+['ensemble_quantile']
    capacity_fields = ['normal_peak_load_gw','required_firm_capacity_gw','available_firm_capacity_gw',
                       'available_transfer_support_gw','adequacy_gap_gw','storage_firm_capacity_gw']
    compared = expanded.merge(ref[keys+capacity_fields], on=keys, suffixes=('_new','_ref'), validate='one_to_one')
    capacity_error = max(float((compared[k+'_new']-compared[k+'_ref']).abs().max()) for k in capacity_fields)
    assert len(compared) == 17856 and capacity_error < 1e-8
    transfer_error = max(abs(event_transfer_support(row,1)-row.available_transfer_support_gw) for row in expanded.itertuples())
    assert transfer_error < 1e-8
    cases = expanded[expanded.risk_level.eq('R50')].set_index(keys)
    expected = pd.read_parquet(ROOT/'reference/operation.parquet')
    matched = pd.read_csv(ROOT/'reference/matched_operation.csv')
    matched = matched[matched.comparison_case.eq('S2_STORAGE_ON_S0_PORTFOLIO')].copy()
    if not args.all_events:
        expected = expected[expected.climate_scenario.eq('ssp245') & expected.ambition_pathway.eq('CN_NDC7_NZ2060') & expected.year.eq(2060)]
        matched = matched[matched.climate_scenario.eq('ssp245') & matched.ambition_pathway.eq('CN_NDC7_NZ2060') & matched.year.eq(2060)]
    profiles = np.load(ROOT/'data/event_profiles.npz', allow_pickle=False)
    corrected.SETTINGS = json.loads((ROOT/'config/reviewer_optimized_design.json').read_text(encoding='utf-8'))['representative_event']
    errors, balance, floors = [], [], []
    for record in expected.to_dict('records'):
        row = cases.loc[tuple(record[k] for k in keys)].to_dict()
        profile = profiles[f"{record['region']}|{record['event_type']}|{record['window_hours']}"]
        outcome = corrected.solve_case(row, row, profile, record['event_type'])
        errors.append(abs(outcome['eens_gwh']-record['eens_gwh']))
        balance.append(max(outcome['max_power_balance_residual_gw'],outcome['max_soc_balance_residual_gwh']))
        floors.append(outcome['minimum_soc_fraction'])
    matched_errors = []
    for record in matched.to_dict('records'):
        common = [record['climate_scenario'],record['ambition_pathway'],'R50']
        base_row = cases.loc[tuple(common+['S0_BASE',record['region'],record['year'],'median'])].to_dict()
        storage = cases.loc[tuple(common+['S2_ACCELERATED',record['region'],record['year'],'median'])].to_dict()
        profile = profiles[f"{record['region']}|{record['event_type']}|{record['window_hours']}"]
        outcome = corrected.solve_case(base_row, storage, profile, record['event_type'])
        matched_errors.append(abs(outcome['eens_gwh']-record['eens_gwh']))
    assert max(errors+matched_errors) < 1e-6 and min(floors) >= .1-1e-7 and max(balance) < 1e-6
    report = {'status':'passed','input_files_verified':len(manifest['files']), 'risk_rows':len(risk),
              'risk_max_error':risk_error,'capacity_rows':len(compared),'capacity_max_error_gw':capacity_error,
              'transfer_identity_max_error_gw':transfer_error,'events_replayed':len(expected),
              'counterfactual_events_replayed':len(matched),'ens_max_error_gwh':max(errors+matched_errors),
              'maximum_balance_residual':max(balance),'minimum_soc_fraction':min(floors),
              'native_gcam_rerun':False,'raw_weather_regenerated':False,
              'scope':'Full risk and capacity reconstruction; dispatch replay from supplied weather-event profiles. Independent operational empirical calibration is not implied.'}
    (ROOT/'reproduction_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
