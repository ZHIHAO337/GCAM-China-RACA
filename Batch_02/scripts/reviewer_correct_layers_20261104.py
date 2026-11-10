from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import analyze_v19_long_term_chain as longterm
import run_v17_duration_event_dispatch as event_base
import run_v19_nested_event_dispatch as dispatch
import run_v20_climate_uncertainty_sensitivity as uncertainty
import run_v24_dynamic_climate_uncertainty as dynamic
from event_supply import event_transfer_support

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('RACA_REVIEW_OUTPUT_ROOT', str(ROOT / 'outputs/reviewer_completion_20261104')))
CALC = OUT / 'calculations'
CAP = CALC / 'v24/dynamic_climate_uncertainty'
EVENT = CALC / 'v24/dynamic_event_dispatch'
MATCH = CALC / 'v24/dynamic_matched_storage_dispatch'
KEYS = ['climate_scenario', 'ambition_pathway', 'risk_level', 'storage_pathway', 'region', 'year']
ALL_YEARS = (2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060)
SENS_YEARS = (2030, 2050, 2060)


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        frame.to_parquet(path, index=False, compression='zstd')
    else:
        frame.to_csv(path, index=False, encoding='utf-8-sig')


def dump(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def capacity():
    for climate in ['ssp245', 'ssp585']:
        source = ROOT / f'outputs/v19/{climate}/processed/province_scenario_metrics.csv'
        target = CALC / f'v19/{climate}/processed/province_scenario_metrics.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/analyze_v4_integrated_adequacy.py'),
                        str(source), '--climate', climate, '--output-dir', str(CALC/f'v19/{climate}/adequacy')], check=True)
    longterm.OUT = CALC / 'v19'
    longterm.ANALYSIS = longterm.OUT / 'long_term_chain'
    longterm.main()
    base = pd.read_csv(CALC/'v19/adequacy/province_capacity_adequacy_with_mechanisms.csv')
    risk = pd.read_csv(ROOT/'data/formal/risk_parameter_uncertainty.csv')
    expanded = uncertainty.build_uncertain_adequacy(base, risk)
    summary = uncertainty.summarize_capacity(expanded)
    firm = expanded[[f'{t}_firm_capacity_gw' for t in ['thermal','hydro','wind','solar','other']]].sum(axis=1)
    error = float((firm-expanded.calibrated_nonstorage_firm_capacity_gw).abs().max())
    assert error < 1e-9
    save(expanded, CAP/'province_capacity_adequacy_uncertainty.parquet')
    save(summary, CAP/'capacity_uncertainty_summary.csv')
    original = pd.read_csv(ROOT/'outputs/v19/adequacy/province_capacity_adequacy_with_mechanisms.csv')
    paired = base.merge(original, on=KEYS, suffixes=('_corrected','_original'), validate='one_to_one')
    metrics = ['adequacy_gap_gw','available_firm_capacity_gw','calibrated_nonstorage_firm_capacity_gw',
               'adequacy_ratio','adequacy_gap_share_required','additional_storage_power_required_gw']
    keep = KEYS.copy()
    for name in metrics:
        paired[name+'_change'] = paired[name+'_corrected']-paired[name+'_original']
        keep += [name+'_original',name+'_corrected',name+'_change']
    save(paired[keep], OUT/'comparison/provincial_capacity_changes.csv')
    dump({'status':'completed','base_rows':len(base),'quantile_rows':len(expanded),
          'single_anchor_identity_max_error_gw':error,'native_gcam_rerun':False},OUT/'capacity_status.json')


def init_worker(records, profiles, settings):
    global RECORDS, PROFILES, SETTINGS
    RECORDS, PROFILES, SETTINGS = records, profiles, settings


def solve_case(row, storage, profile, event_type):
    load_shape, wind_shape, solar_shape = profile
    mult = event_base.EVENT_MULTIPLIERS[event_type]
    load = load_shape*row['normal_peak_load_gw']*(1+row['delta_peak']*mult['peak'])
    supply = (
        row['thermal_anchored_capacity_gw']*.9*(1-row['derate_thermal']*mult['thermal'])
        + row['hydro_anchored_capacity_gw']*.5*(1-row['derate_hydro']*mult['hydro'])
        + row['other_anchored_capacity_gw']*.5
        + row['wind_anchored_capacity_gw']*wind_shape*(1-row['derate_wind']*mult['wind'])
        + row['solar_anchored_capacity_gw']*solar_shape*(1-row['derate_pv']*mult['pv'])
        + event_transfer_support(row, mult['trade'])
    )
    return dispatch.dispatch_with_soc(np.asarray(load), np.asarray(supply), storage['storage_power_gw'],
        storage['storage_energy_gwh'], SETTINGS['round_trip_efficiency'],
        SETTINGS['initial_storage_soc_fraction'], SETTINGS['terminal_storage_soc_fraction'],
        minimum_fraction=float(SETTINGS.get('minimum_storage_soc_fraction', 0.10)))


def worker(task):
    mode, quantile, event = task
    hours, province, event_type = event['window_hours'], event['province_code'], event['event_type']
    profile = PROFILES[(province, event_type, hours)]
    years = ALL_YEARS if hours == 168 else SENS_YEARS
    rows = [r for r in RECORDS[(quantile,province)] if r['year'] in years]
    result = []
    storage_lookup = {(r['climate_scenario'],r['ambition_pathway'],r['year']):r for r in rows if r['storage_pathway']=='S2_ACCELERATED'}
    for row in rows:
        if mode == 'matched' and row['storage_pathway'] != 'S0_BASE':
            continue
        storage = row if mode == 'ordinary' else storage_lookup[(row['climate_scenario'],row['ambition_pathway'],row['year'])]
        outcome = solve_case(row, storage, profile, event_type)
        record = {k:row[k] for k in KEYS}
        record.update({k:event[k] for k in ['episode_id','window_hours','event_type','event_start','event_end']})
        record.update(outcome)
        record['ensemble_quantile'] = quantile
        if mode == 'matched':
            record['comparison_case'] = 'S2_STORAGE_ON_S0_PORTFOLIO'
        result.append(record)
    return mode, result


def operations(workers):
    started = time.time()
    adequacy = pd.read_parquet(CAP/'province_capacity_adequacy_uncertainty.parquet')
    subset = adequacy[adequacy.risk_level.eq('R50')]
    records = {k:g.to_dict('records') for k,g in subset.groupby(['ensemble_quantile','region'])}
    events = pd.read_csv(ROOT/'outputs/v24/dynamic_event_dispatch/selected_nested_events.csv',
                         parse_dates=['event_start','event_end','parent_336_start','parent_336_end'])
    scored = event_base.build_scores(event_base.load_era5())
    profiles = {}
    for event in events.itertuples(index=False):
        weather = scored[scored.province_code.eq(event.province_code)&scored.date.between(event.event_start,event.event_end)].sort_values('date')
        profiles[(event.province_code,event.event_type,event.window_hours)] = event_base.hourly_profiles(weather,event.event_type)
    settings = json.loads((ROOT/'config/reviewer_optimized_design.json').read_text(encoding='utf-8'))['representative_event']
    tasks = [('ordinary',q,e) for q in ['median','p10','p90'] for e in events.to_dict('records')]
    tasks += [('matched','median',e) for e in events.to_dict('records')]
    all_rows, matched_rows = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(records,profiles,settings)) as pool:
        for i,(mode,result) in enumerate(pool.map(worker,tasks,chunksize=1),1):
            (all_rows if mode=='ordinary' else matched_rows).extend(result)
            if i % 20 == 0 or i == len(tasks):
                progress={'status':'running','groups_done':i,'groups_total':len(tasks),
                          'dispatch_rows':len(all_rows),'matched_new_rows':len(matched_rows),
                          'elapsed_seconds':time.time()-started}
                dump(progress,OUT/'dispatch_progress.json')
                print(json.dumps(progress),flush=True)
    result = pd.DataFrame(all_rows)
    assert len(result)==117180
    assert result.minimum_soc_fraction.min() >= .1-1e-7
    assert result.maximum_soc_fraction.max() <= 1+1e-7
    assert result.terminal_soc_fraction.min() >= settings['terminal_storage_soc_fraction']-1e-7
    assert result.max_soc_balance_residual_gwh.max()<1e-6
    assert result.max_power_balance_residual_gw.max()<1e-6
    save(result,CAP/'province_event_dispatch_uncertainty.parquet')
    provinces, nationals = [], []
    for q, group in result.groupby('ensemble_quantile'):
        p,n,e = dispatch.summarize(group)
        p['ensemble_quantile'], n['ensemble_quantile'] = q,q
        provinces.append(p); nationals.append(n)
        if q == 'median':
            save(group.drop(columns='ensemble_quantile'),EVENT/'province_event_duration_reliability.csv')
            save(p.drop(columns='ensemble_quantile'),EVENT/'province_duration_reliability.csv')
            save(n.drop(columns='ensemble_quantile'),EVENT/'duration_reliability_summary.csv')
            save(e,EVENT/'event_type_reliability_summary.csv')
            save(events,EVENT/'selected_nested_events.csv')
    province, national = pd.concat(provinces,ignore_index=True),pd.concat(nationals,ignore_index=True)
    operation = uncertainty.summarize_operation(province)
    capacity_summary = pd.read_csv(CAP/'capacity_uncertainty_summary.csv')
    for frame,name in [(province,'province_operation_uncertainty'),(national,'national_operation_uncertainty'),
                       (operation,'operation_uncertainty_summary'),
                       (dynamic.dynamic_effects(capacity_summary,operation),'storage_effect_uncertainty')]:
        save(frame,CAP/(name+'.csv'))
    matched = pd.DataFrame(matched_rows)
    assert len(matched)==13020
    reused = result[result.ensemble_quantile.eq('median')&result.storage_pathway.eq('S0_BASE')].copy()
    reused['comparison_case'] = 'S0_MATCHED'
    matched = pd.concat([reused,matched],ignore_index=True).drop(columns=['storage_pathway','ensemble_quantile'])
    save(matched,MATCH/'matched_storage_event_reliability.csv')
    paired_keys = ['climate_scenario','ambition_pathway','region','year','window_hours','event_type']
    effect = matched.pivot(index=paired_keys,columns='comparison_case',values='eens_gwh').reset_index()
    effect['direct_storage_eens_reduction_gwh']=effect.S0_MATCHED-effect.S2_STORAGE_ON_S0_PORTFOLIO
    assert effect.direct_storage_eens_reduction_gwh.min()>=-1e-6
    save(effect,MATCH/'matched_storage_direct_effects.csv')
    original=pd.read_parquet(ROOT/'outputs/v24/dynamic_climate_uncertainty/province_event_dispatch_uncertainty.parquet')
    compare_keys=KEYS+['ensemble_quantile','window_hours','event_type']
    paired=result.merge(original[compare_keys+['eens_gwh','minimum_soc_fraction','hours_at_soc_floor']],on=compare_keys,
                        suffixes=('_corrected','_original'),validate='one_to_one')
    paired['ens_change_gwh']=paired.eens_gwh_corrected-paired.eens_gwh_original
    # SOC tightening alone is monotonic; a transfer-interface correction need not be.
    save(paired[compare_keys+['eens_gwh_original','eens_gwh_corrected','ens_change_gwh',
        'minimum_soc_fraction_original','minimum_soc_fraction_corrected','hours_at_soc_floor_original','hours_at_soc_floor_corrected']],
         OUT/'comparison/operation_changes.parquet')
    audit=uncertainty.audit_results(adequacy,capacity_summary,result,operation)
    audit.update({'status':'completed','new_lp_solves':len(result)+13020,'matched_pairs':len(effect),
                  'soc_floor':.10,'minimum_soc_fraction':float(result.minimum_soc_fraction.min()),
                  'maximum_soc_fraction':float(result.maximum_soc_fraction.max()),
                  'max_power_balance_residual_gw':float(result.max_power_balance_residual_gw.max()),
                  'max_soc_balance_residual_gwh':float(result.max_soc_balance_residual_gwh.max()),
                  'ens_increased_cases':int((paired.ens_change_gwh>1e-6).sum()),
                  'maximum_ens_increase_gwh':float(paired.ens_change_gwh.max()),
                  'negative_direct_effect_cases':int((effect.direct_storage_eens_reduction_gwh < -1e-6).sum()),
                  'transfer_portfolio_consistent_with_capacity':True,
                  'elapsed_seconds':time.time()-started})
    dump(audit,OUT/'dispatch_status.json')
    dump(audit,CAP/'climate_uncertainty_audit.json')
    dump({'status':'completed',**audit},OUT/'dispatch_progress.json')
    print(json.dumps(audit,indent=2),flush=True)


def derived():
    import prepare_v29_three_path_results as results
    import run_v29_core_capacity_sensitivity as sensitivity
    results.OUT=CALC/'v29_three_path_results'
    results.FIG=results.OUT/'figures'; results.TABLE=results.OUT/'tables'
    results.FIG.mkdir(parents=True,exist_ok=True)
    results.CAPACITY_FILE=CAP/'capacity_uncertainty_summary.csv'
    results.OPERATION_FILE=CAP/'operation_uncertainty_summary.csv'
    results.EVENT_FILE=EVENT/'province_event_duration_reliability.csv'
    results.DURATION_FILE=EVENT/'province_duration_reliability.csv'
    results.ADEQUACY_FILE=CALC/'v19/adequacy/province_capacity_adequacy_with_mechanisms.csv'
    results.LONG_FILE=CALC/'v19/long_term_chain/national_long_term_summary.csv'
    results.DIRECT_FILE=MATCH/'matched_storage_direct_effects.csv'
    results.main()
    sensitivity.ADEQUACY_PATH=CAP/'province_capacity_adequacy_uncertainty.parquet'
    sensitivity.CAPACITY_SUMMARY_PATH=CAP/'capacity_uncertainty_summary.csv'
    sensitivity.OUTPUT=CALC/'v29_core_capacity_sensitivity'
    sensitivity.main()
    dump({'status':'completed','v29_three_path_results':str(results.OUT),
          'core_capacity_sensitivity':str(sensitivity.OUTPUT)},OUT/'derived_status.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['capacity','dispatch','derived','all'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage in ['capacity','all']: capacity()
    if args.stage in ['dispatch','all']: operations(args.workers)
    if args.stage in ['derived','all']: derived()
