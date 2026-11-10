# GCAM-China-RACA

Code, scenario inputs and numerical results for provincial power-system resilience assessment under climate-ambition and storage-transition pathways in China.

Repository: https://github.com/ZHIHAO337/GCAM-China-RACA

## Reproduce the numerical assessment

Use Python 3.12 and the dependency versions in `requirements.txt`. From the repository root:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The default run verifies the archived file checksums, reconstructs 5,952 climate-risk records and 17,856 capacity records, and replays 4,185 dispatch cases and 465 storage counterfactuals for the 2060 NDC-7 / SSP2-4.5 subset. It checks the results against the supplied reference tables, including power balance, storage-energy balance and the continuous 10% state-of-charge floor. The numerical check is written to `reproduction_check.json`.

To replay all 117,180 dispatch cases and 13,020 storage counterfactuals:

```bash
python reproduce.py --all-events
```

The full replay takes longer than the default check. The reference dispatch table contains three climate-ensemble quantiles for 39,060 province-scenario-event-duration units.

## Files

| Location | Contents |
| --- | --- |
| `reproduce.py` | Supported entry point for risk and capacity reconstruction and event-window dispatch replay. |
| `config/` | Risk assumptions, prescribed storage pathways and event-dispatch settings. |
| `data/formal/` | Derived historical and future climate-risk inputs. |
| `data/processed/` | Provincial descriptors and inputs to the storage and technology-derating implementation. |
| `data/event_profiles.npz` | Derived event profiles used by the dispatch replay. |
| `data/selected_nested_events.csv` | Selected event dates and durations. |
| `data/v4/` | Climate-ambition constraints and the 24-scenario input manifest. |
| `outputs/` | Archived long-term portfolio interfaces used in the reconstruction. |
| `reference/` | Revised capacity, operational and matched-counterfactual results used for numerical verification. |
| `scripts/` | Risk, capacity, dispatch and result-processing code, plus the study-specific native scenario-input builders. |
| `model_xml/v4_integrated/` | The 24 study-specific GCAM scenario components and their batch/configuration files. |
| `queries/` | Native-model query definitions. |
| `evidence/` | Implementation, parameter-provenance and validation records. |
| `MODEL_IMPLEMENTATION.md` | Implementation responsibilities and native-model dependencies. |
| `manifest.json` | File checksums and the recorded numerical environment. |

## Coverage and interpretation

The long-term design contains three climate-ambition pathways, two physical-climate backgrounds and four risk-storage settings, for 24 scenarios. The 168-h analysis covers 2025, 2030, 2035, 2040, 2045, 2050, 2055 and 2060; the 72-h and 336-h analyses cover 2030, 2050 and 2060.

ENS denotes deterministic energy not served within a specified event window. Some legacy column names retain `eens_gwh`; they contain this event-window ENS quantity. Province-wise worst-event sums aggregate separate provincial stress tests. The physical-climate backgrounds are used to test portfolios under alternative climate conditions.

`reproduce.py` starts from the supplied long-term portfolio interfaces and derived weather-event profiles. The native model used GCAM 8.2 (`gcam-v8.2`); the study-specific XML and implementation source are supplied separately for inspection. A new native-model solve requires the corresponding GCAM-China installation and base inputs. Raw ERA5-Land and NEX-GDDP-CMIP6 grids, the model executable and database are obtained separately from their providers. See `MODEL_IMPLEMENTATION.md` for the distinction between the native inputs and the supported numerical replay.

## Source provenance

The formal risk builder uses `risk_levels` and `baseline_reserve_margin` from `config/model_assumptions.json`. Active storage settings come from `config/reviewer_optimized_design.json`; future climate inputs come from the supplied ERA5-Land/CMIP6-derived tables. Legacy storage and proxy-amplification fields in the assumptions file are retained for provenance.

The `evidence/initial_audit_model_layer_responsibilities.csv` file records an earlier audit state. The current responsibility table is `evidence/model_layer_responsibilities.csv`.

The original GCAM source is maintained by [JGCRI](https://github.com/JGCRI/gcam-core/tree/gcam-v8.2). Derived inputs and study-specific files are distinguished from upstream model and climate-data distributions in this repository.
