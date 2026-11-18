# Paper-to-code map

| Reported analysis | Module and command | Notes |
|---|---|---|
| Table 1 reference-dependent industry-year accounting | `reference_decomposition`, `sparse_cells`; `core` for main reference diagnostics | Canonical panel; sparse-cell thresholds available through API |
| Fig. 1 / Table 2 allocation and regimes | `baseline.exposure_tables`, `diagnostics.build_core_gtos_regime`; `core` | Canonical panel with stored baseline fields |
| Fig. 2 / Table 3 nested coefficients and Gelbach | `baseline`, `decomposition`; `core` | Final common-sample rule |
| Table 4 pooled Oaxaca accounting | `decomposition`; `core` | Pooled coefficients exclude agricultural indicator |
| Table 5 entry, positive-count intensity and interactions | `count_models.run_two_part`; `core` | Original entry/intensity samples, original post-2015 definition |
| Table 6 ranges, reassignment, clustering, count models | `support`, `baseline`, `diagnostics`, `count_models`; `core` / `reassignment` | Marginal range check, not multivariate common support; frequencies are heuristic |
| A1 / A3 sample flow and support | `sample`; `sample` | Sector A classification summaries are available through the `sector_a` Python module |
| A3 Panel G sample-restriction comparison | `sample_bridge`; `sample-bridge` | Post-2007 static-complete sample versus the common sample; then add stored GTOS on identical rows; point estimates only |
| A2 industry-FE baseline | `industry_baseline_models`; `supplementary` | Patent outcomes with the reported complete-case rules |
| A5 contextual penalty comparisons | `penalties`; `penalties` | Complete source years 2009-2023; specification-specific samples in Panel A and 42,332 original common-sample rows in Panels B-C |
| A2 multicollinearity, FWL, coefficient paths | `sign_diagnostics`; VIFs and condition indices in `core`, other diagnostics through the Python API | Uses the common estimation sample |
| A2 alternative controls and industry-FE role | `controls`; `supplementary` | Original revenue/employment fields and R&D export required; preparation follows original calendar lag and control caps |
| A7 Panel F industry-code source sensitivity | `industry_sources`; `supplementary` | Primary CSMAR field and fallback field from the green credit compilation |
| A2 bootstrap uncertainty | `accounting`; `bootstrap` | 2,000 firm draws, seed 20261008; no redraw of failed samples; proxies held fixed |
| A2 jointly estimated NB2 | `nb2`; `nb2` | Original starts, optimizer tolerances, likelihood and cluster-score implementation |
| A2 own-history comparison | `history`, `rd`; `history` | Exact calendar own lag; same R&D/history intersection |
| A2 leave-industry-out / shared industries | `influence`, `shared_industries`; corresponding commands | Stored GTOS retained; no rebuilding from deletion samples |
| A2 Panel N finite-cluster inference | `wild_cluster`; `wild-cluster` | Fourteen fixed coefficient tests; original common, positive-count, R&D/history and guide samples; requires panel, R&D export and bundled guide coding |
| A4 alternative clustering / late window | `diagnostics`; `core` | Firm, industry and two-way schemes from original routines |
| A6-A8 reference, support and tail checks | `descriptive_alignment`, `reference_decomposition`, `sparse_cells`, `diagnostics`; `descriptive`, `supplementary`, `tails` | A7 uses fixed caps of 16, 32 and 83. The high-activity bootstrap uses seed 20260728. |
| A9 R&D intensity and personnel comparisons | `rd`; `rd` | Requires the CSMAR R&D export |
| A9 Panel A R&D and investment comparisons | `rd_triangulation`, `investment_triangulation`; `supplementary` | Uses the CSMAR R&D and Macrodatas investment exports |
| A9 expense/capital comparison | `investment`, `environment`; `investment` | Two original reported ratios, separate samples |
| A9 subsidy title share / detail title shares | `subsidy`, `projects`, `investment`, `rd`, `history`; corresponding commands | Requires the original project-title classifications |
| A9 guide profiles | `guide_design`, `guides`; `guide` | Bundled author coding, 2018-2020 role-separated fixed samples |

## Running analyses

Run commands from the repository root. See the [input schema](INPUT_SCHEMA.md) for the required files. Write outputs to new JSON files outside the repository:

```bash
python run.py rd --panel ../licensed/panel.csv --rd ../licensed/PT_LCRDSPENDING.xlsx --output ../results/rd.json
```

Use `python run.py COMMAND --help` for options. The `subsidy` and `projects` commands also need restricted project-title classifications.

Use `descriptive` for the reported A6/A7 tail checks. It uses the fixed caps above instead of estimating percentiles on each subset.

`reassignment` uses 500 firm-label, 500 within-cell and 200 GTOS-bundle reassignments.

`penalties` compares 2009-2014 with 2015-2023. It excludes years without complete penalty coverage and keeps the full-panel winsorization cutoffs.

`sample-bridge` separates changing sample membership from adding GTOS; it is not a selection correction. `wild-cluster --rd PATH` tests individual agricultural coefficients using restricted wild industry-cluster bootstrap-t (WCR11). It retains the original 14-test order and seed 20261028, with 9,999 Rademacher and Webb draws and exhaustive 8,192 Rademacher signs for 13 industries. It does not test coefficient changes or decomposition components. Neither command runs in `check`.

The leave-one-out helper uses the preceding firm row. It requires consecutive firm years, including when called directly through Python; the CLI checks this before selecting the sample.

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

It compares controls, patent GTOS and the non-patent input in separate specifications and repeats the comparisons on the sample with observed prior-year patent histories. The analysis requires the original project-title labels.
