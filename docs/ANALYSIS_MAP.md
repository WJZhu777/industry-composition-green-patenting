# Paper-to-code map

The package includes retained numerical functions from the final analytical sources, not all historical project scripts. `provenance.json` records each function's original file and AST checksum. Release-only input and output checks have separate checksums where they change a retained function. Document synchronization, source-screening experiments not used in the paper, old IV/policy routes and Word editing are excluded.

| Reported analysis | Portable module / entrypoint | Input / status |
|---|---|---|
| Table 1 reference-dependent industry-year accounting | `reference_decomposition`, `sparse_cells`; `core` for main reference diagnostics | Canonical panel; sparse-cell thresholds available through API |
| Fig. 1 / Table 2 allocation and regimes | `baseline.exposure_tables`, `diagnostics.build_core_gtos_regime`; `core` | Canonical panel with stored baseline fields |
| Fig. 2 / Table 3 nested coefficients and Gelbach | `baseline`, `decomposition`; `core` | Final common-sample rule |
| Table 4 pooled Oaxaca accounting | `decomposition`; `core` | Pooled coefficients exclude agricultural indicator |
| Table 5 entry, positive-count intensity and interactions | `count_models.run_two_part`; `core` | Original entry/intensity samples, original post-2015 definition |
| Table 6 ranges, reassignment, clustering, count models | `support`, `baseline`, `diagnostics`, `count_models`; `core` / `reassignment` | Marginal range check, not multivariate common support; frequencies are heuristic |
| A1 / A3 sample flow and support | `sample`, `sector_a`; `sample` | Panel plus classification/source fields for sector diagnostics |
| A3 Panel G sample-restriction comparison | `sample_bridge`; `sample-bridge` | Post-2007 static-complete sample versus the common sample; then add stored GTOS on identical rows; point estimates only |
| A2 industry-FE baseline | `industry_baseline_models`; `supplementary` | Original patent outcomes and complete-case rules; no legacy panel builder |
| A5 contextual penalty comparisons | `penalties`; `penalties` | Complete source years 2009-2023; specification-specific samples in Panel A and 42,332 original common-sample rows in Panels B-C |
| A2 multicollinearity, FWL, coefficient paths | `sign_diagnostics`; VIF/condition-index part in `core`, remaining functions via API | Fits are not cached across calls or samples; repeated order specifications reuse fits only within one call |
| A2 alternative controls and industry-FE role | `controls`; `supplementary` | Original revenue/employment fields and R&D export required; preparation follows original calendar lag and control caps |
| A7 Panel F industry-code source sensitivity | `industry_sources`; `supplementary` | Primary CSMAR field and fallback field from the green credit compilation |
| A2 bootstrap uncertainty (P3) | `accounting`; `bootstrap` | 2,000 firm draws, seed 20261008; no redraw of failed samples; proxies held fixed |
| A2 jointly estimated NB2 (P4) | `nb2`; `nb2` | Original starts, optimizer tolerances, likelihood and cluster-score implementation |
| A2 own-history comparison (P8) | `history`, `rd`; `history` | Exact calendar own lag; same R&D/history intersection |
| A2 leave-industry-out / shared industries (P13/P15) | `influence`, `shared_industries`; corresponding commands | Stored GTOS retained; no rebuilding from deletion samples |
| A2 Panel N finite-cluster inference | `wild_cluster`; `wild-cluster` | Fourteen fixed coefficient tests; original common, positive-count, R&D/history and guide samples; requires panel, R&D export and bundled guide coding |
| A4 alternative clustering / late window | `diagnostics`; `core` | Firm, industry and two-way schemes from original routines |
| A6-A8 reference/support/tail diagnostics | `descriptive_alignment`, `reference_decomposition`, `sparse_cells`, `diagnostics`; `descriptive`, `supplementary`, `tails` | Final A7 tail caps are 16/32/83, not recomputed within the valid-code subset; high-regime bootstrap uses its original group-specific seed 20260728 |
| A9 R&D intensity and personnel comparisons | `rd`; `rd` | Requires the CSMAR R&D export |
| A9 earlier R&D/investment construct comparisons | `rd_triangulation`, `investment_triangulation`; `supplementary` | Original source schemas required; zero/missing and denominator rules retained |
| A9 expense/capital comparison (P11) | `investment`, `environment`; `investment` | Two original reported ratios, separate samples |
| A9 subsidy title share / detail title shares (P18/P20) | `subsidy`, `projects`, `investment`, `rd`, `history`; corresponding commands | Construct/audit/compare; frozen labels remain restricted |
| A9 guide profiles (P31-P33) | `guide_design`, `guides`; `guide` | Bundled author coding, 2018-2020 role-separated fixed samples |

## Running analyses

Run commands from the repository root with the inputs specified in `INPUT_SCHEMA.md`. Empirical outputs must be new JSON files outside the repository:

```bash
python run.py validate-inputs --panel ../licensed/panel.csv
python run.py sample --panel ../licensed/panel.csv --output ../results/sample.json
python run.py core --panel ../licensed/panel.csv --output ../results/core.json
python run.py penalties --panel ../licensed/panel.csv --output ../results/penalties.json
python run.py guide --panel ../licensed/panel.csv --output ../results/guide.json
python run.py rd --panel ../licensed/panel.csv --rd ../licensed/PT_LCRDSPENDING.xlsx --output ../results/rd.json
```

Other commands are `history`, `investment`, `subsidy`, `projects`, `bootstrap`, `nb2`, `influence`, `shared`, `reassignment`, `tails`, `supplementary`, `descriptive`, `sample-bridge` and `wild-cluster`. Use `python run.py COMMAND --help` for required files. The `subsidy` and `projects` commands need frozen classification ledgers as well as source exports; those ledgers are not included.

Use `descriptive` for the final A6/A7 tail tables. It retains the reported caps of 16, 32 and 83; the generic within-subset percentile helper is not equivalent. The `supplementary` command covers alternative controls, industry-code sources and earlier R&D/investment comparisons.

The `bootstrap` command uses 2,000 firm draws and the original seed. The `reassignment` command uses 500 firm-label, 500 within-cell and 200 GTOS-bundle reassignments. Neither runs as part of `check`.

`penalties` reproduces all three panels of Table A5. The export begins in 2009 and ends on 30 April 2024, so penalty models exclude 2007-2008 and incomplete 2024. The period split is 2009-2014 versus 2015-2023. The legacy baseline API applies the same restriction to penalty outcomes only. Patent samples and the original full-panel winsorization cutoffs are unchanged.

`sample-bridge` separates changing sample membership from adding GTOS; it is not a selection correction. `wild-cluster --rd PATH` tests individual agricultural coefficients using restricted wild industry-cluster bootstrap-t (WCR11). It retains the original 14-test order and seed 20261028, with 9,999 Rademacher and Webb draws and exhaustive 8,192 Rademacher signs for 13 industries. It does not test coefficient changes or decomposition components. Neither command runs in `check`.

Run `validate-inputs` before estimation. The legacy leave-one-out helper uses row shifts for each firm's own history and requires consecutive firm years; the canonical panel satisfies this condition, and the CLI rejects gaps before sample selection. Direct API callers must also supply complete firm histories, not a gapped or prefiltered panel.

## Display labels

The paper uses `Low activity`, `Middle activity` and `High activity`. Saved code outputs retain `Low-opportunity`, `Middle-opportunity` and `High/frontier opportunity`, respectively. Group membership and calculations are unchanged.

Table 6 uses `Mean under joint GTOS-profile reassignment` for the simulation retained in code and saved results under `GTOS-bundle placebo`. This is a display-name change, not a different procedure.

The Oaxaca output label `Explained total` denotes the paper's `model-accounted component`. It is an algebraic quantity, not a causal explanation or a share of variance explained.

## Supplementary title-share regression API

The `subsidy` and `projects` commands call the same routine used for the original reported regressions. With a correctly prepared external `merged` panel and the original indicator `variable`, its API is:

```python
from research_release import investment
result = investment.compare(merged, variable, label)
```

It compares controls, patent GTOS and the non-patent input in separate specifications; it repeats them on the exact own-history intersection. This routine was used in P18/P20. Construction depends on the original frozen commercial-record labels, not a newly selected keyword screen.

## Remaining integration limits

- The public package begins at the canonical analysis panel; it is not a complete raw-provider export builder.
- Alternative controls and earlier A9 comparisons now have an explicit `supplementary` command. The complete raw-source build and frozen restricted labels remain outside the public package.
- `core` and all other explicit empirical commands were run using the authorized original inputs. Main results were compared with selected current saved results and displayed tables. The generic `core` tail-percentile helper is not the final A7 table; `descriptive` implements that table's original fixed caps.
- No Word, figure or submission-file generator is included. Reference table extraction is a snapshot, not model execution.
- The full component bootstrap, count models and final guide regressions were rerun. [Reproduction notes](RELEASE_CHECKLIST.md#reproduction-notes) document NB2 floating-point differences and Poisson threading sensitivity.
- Some supplementary functions remain API-only, including parts of the FWL/coefficient-path diagnostics and sector-A feasibility summaries. This run does not certify every historical helper or every displayed table cell.
