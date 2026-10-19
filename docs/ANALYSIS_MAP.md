# Paper-to-code map

The package includes retained numerical functions from the final analytical sources, not all historical project scripts. `provenance.json` records each function's original file and AST checksum. Document synchronization, source-screening experiments not used in the paper, old IV/policy routes and Word editing are excluded.

| Reported analysis | Portable module / entrypoint | Input / status |
|---|---|---|
| Table 1 reference-dependent industry-year accounting | `reference_decomposition`, `sparse_cells`; `core` for main reference diagnostics | Canonical panel; sparse-cell thresholds available through API |
| Fig. 1 / Table 2 allocation and regimes | `baseline.exposure_tables`, `diagnostics.build_core_gtos_regime`; `core` | Canonical panel with stored baseline fields |
| Fig. 2 / Table 3 nested coefficients and Gelbach | `baseline`, `decomposition`; `core` | Final common-sample rule |
| Table 4 pooled Oaxaca accounting | `decomposition`; `core` | Pooled coefficients exclude agricultural indicator |
| Table 5 entry, positive-count intensity and interactions | `count_models.run_two_part`; `core` | Original entry/intensity samples, original post-2015 definition |
| Table 6 ranges, reassignment, clustering, count models | `support`, `baseline`, `diagnostics`, `count_models`; `core` / `reassignment` | Marginal range check, not multivariate common support; frequencies are heuristic |
| A1 / A3 sample flow and support | `sample`, `sector_a`; `sample` | Panel plus classification/source fields for sector diagnostics |
| A2/A5 original industry-FE baseline and penalty context | `industry_baseline_models`; `supplementary` | Original outcome fields and complete-case rules; selected reported A5 entries checked at displayed precision; no legacy panel builder |
| A2 multicollinearity, FWL, coefficient paths | `sign_diagnostics`; VIF/condition-index part in `core`, remaining functions via API | Fits are not cached across calls or samples; repeated order specifications reuse fits only within one call |
| A2 alternative controls and industry-FE role | `controls`; `supplementary` | Original revenue/employment fields and R&D export required; preparation follows original calendar lag and control caps |
| A2 classification-source sensitivity | `industry_sources`; `supplementary` | Original dual-source fields required; no alternate classifications invented |
| A2 bootstrap uncertainty (P3) | `accounting`; `bootstrap` | 2,000 firm draws, seed 20261008; no redraw of failed samples; proxies held fixed |
| A2 jointly estimated NB2 (P4) | `nb2`; `nb2` | Original starts, optimizer tolerances, likelihood and cluster-score implementation |
| A2 own-history comparison (P8) | `history`, `rd`; `history` | Exact calendar own lag; same R&D/history intersection |
| A2 leave-industry-out / shared industries (P13/P15) | `influence`, `shared_industries`; corresponding commands | Stored GTOS retained; no rebuilding from deletion samples |
| A4 alternative clustering / late window | `diagnostics`; `core` | Firm, industry and two-way schemes from original routines |
| A6-A8 reference/support/tail diagnostics | `descriptive_alignment`, `reference_decomposition`, `sparse_cells`, `diagnostics`; `descriptive`, `supplementary`, `tails` | Final A7 tail caps are 16/32/83, not recomputed within the valid-code subset; high-regime bootstrap uses its original group-specific seed 20260728 |
| A9 new R&D/personnel comparison (P2) | `rd`; `rd` | Separate private CSMAR input |
| A9 earlier R&D/investment construct comparisons | `rd_triangulation`, `investment_triangulation`; `supplementary` | Original source schemas required; zero/missing and denominator rules retained |
| A9 expense/capital comparison (P11) | `investment`, `environment`; `investment` | Two original reported ratios, separate samples |
| A9 subsidy title share / detail title shares (P18/P20) | `subsidy`, `projects`, `investment`, `rd`, `history`; corresponding commands | Construct/audit/compare; frozen labels remain restricted |
| A9 guide profiles (P31-P33) | `guide_design`, `guides`; `guide` | Bundled author coding, 2018-2020 role-separated fixed samples |

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
- The full component bootstrap, count models and final guide regressions were rerun. NB2 floating-point differences and Poisson threading sensitivity are documented in the README; exact numerical identity is not claimed.
- Some supplementary functions remain API-only, including parts of the FWL/coefficient-path diagnostics and sector-A feasibility summaries. This run does not certify every historical helper or every displayed table cell.

Do not describe this candidate as complete open-data, end-to-end reproduction before these limits and supplier access conditions are resolved or clearly disclosed in the public release.
