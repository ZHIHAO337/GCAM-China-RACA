# Study-specific model implementation

## Long-term scenario inputs

The native run logs identify GCAM 8.2, revision `gcam-v8.2`; the recorded version lines are in `evidence/model_version_evidence.csv`.

The scenario-input code is provided alongside the archived scenario XML:

- `scripts/build_reviewer_optimized_raca.py` specifies storage deployment and generation-availability adjustments. `build_storage_targets` allocates prescribed national storage power using grid-region shares and within-grid provincial electricity-consumption shares. The duration mix determines storage energy. `add_global_storage_technologies` and `add_storage` write storage technology parameters and annual reference output to the native XML.
- In the same script, `applied_derate` and `add_derates` apply technology-specific weather adjustments to capacity-factor or fixed-output fields. Peak-generation segments receive the conditional derating; other segments use the configured duration weighting, with a separate hydro-drought duration.
- `scripts/build_v4_integrated_ambition.py` adds the China-wide emissions-policy constraints and builds the crossed ambition-risk-storage scenario components.
- `scripts/postprocess_v4_integrated.py` and `scripts/postprocess_raca.py` document the native-query interface. Prescribed storage power and energy are retained separately from native annual storage output.

`data/v4/integrated_scenario_manifest.json` identifies all 24 components in `model_xml/v4_integrated/`. The accompanying configuration and batch files record the original native-run layout. They refer to the GCAM-China base inputs under `../input/`, and to the study directory under `../gcam_raca/`, relative to the native executable directory. The native query definitions are in `queries/raca_v4_integrated_queries.xml`.

These source builders retain their original project-directory assumptions and auxiliary-input dependencies. The supplied final scenario XML is the inspection record for the native inputs; the portable reproduction command is `python reproduce.py`, not a builder invocation. Rebuilding a native model database requires the compatible GCAM-China base distribution and its input data in addition to these study-specific files. The upstream GCAM source tag is [gcam-v8.2](https://github.com/JGCRI/gcam-core/tree/gcam-v8.2).

## Risk-adjusted capacity and event dispatch

The formal climate-risk transformation is implemented in `scripts/build_formal_climate_inputs.py`. Peak-load changes and interprovincial support losses enter the capacity and dispatch assessments. Generation-availability adjustments are also present in the upstream native scenario XML, as described above.

The supported replay reconstructs the capacity assessment from archived provincial portfolios, using the revised capacity accounting and the same prescribed storage power and energy. The dispatch solver applies a continuous minimum state of charge of 10% and a separate terminal-state condition. Interprovincial support is transferred consistently from the capacity portfolio before applying the event-specific loss multiplier.

`evidence/model_layer_responsibilities.csv` summarizes the implemented responsibility of each model layer. `evidence/storage_actual_vs_prescribed_output.csv` distinguishes prescribed storage power/energy from solved annual output. The parameter-provenance records preserve the original assumptions-file checksum and its numerical verification.

## Reproduction boundary

The default and full-replay commands compare reconstructed risk, capacity and event-dispatch quantities with the archived revised numerical references. They use the supplied derived hazard tables and event profiles. Regeneration of raw meteorological fields and a new native GCAM-China solve are separate workflows with upstream software and data dependencies.

Historical input records, scenario code and revised numerical references have distinct roles. The reference files used by `reproduce.py` are `reference/capacity.parquet`, `reference/operation.parquet` and `reference/matched_operation.csv`.
