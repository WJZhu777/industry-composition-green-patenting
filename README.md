# Industry composition and green invention patenting

Project name: `industry-composition-green-patenting`. The local release directory is `10_industry_composition_green_patenting`; it is separate from the original research workspace and local Python/IDE environments.

Research code and public-source classifications accompanying **Industry Composition and the Green Invention Patenting Gap between Agricultural and Other Listed Firms in China**, submitted to *Environment, Development and Sustainability*.

**Status:** private GitHub repository; not publicly released. No public license has been selected, and no Zenodo archive has been created.

Repository: [WJZhu777/industry-composition-green-patenting](https://github.com/WJZhu777/industry-composition-green-patenting).

This package contains analysis code, public-source classifications and aggregate reference tables. Empirical reproduction requires the restricted inputs described below; the package does not rebuild the analysis panel from raw database exports.

## Contents

```text
run.py                         Explicit analysis and checking entrypoint
requirements.txt               Versions in the original analytical environment
code/research_release/         Portable numerical and diagnostic functions
data/public/                   Author-created guide coding and official source links
results/reference/             Displayed aggregate tables from the current paper
tests/                         Synthetic and public-coding checks
docs/                          Input contracts, analysis mapping and release conditions
provenance.json                Original source files and retained-function checksums
```

The analysis is descriptive and non-causal. Patent-based GTOS variables describe observed industry-year patent activity. Accounting partitions, coefficient changes and the guide comparison do not identify an exogenous opportunity effect or an agricultural advantage.

## What is and is not included

- Included: analysis code, author-created conditional industry/role classifications for seven official 2017 R&D guides, source links, and the paper's aggregate tables.
- Excluded commercial inputs: CNRDS patent records; CCAD (Zhejiang University-Qiyan) agricultural-firm membership; CSMAR company attributes, R&D, environmental investment and green subsidies; CCER environmental penalties. Their firm-level panels, matching records, industry-year series and row-linked title labels are also excluded.
- The earlier 1999-2024 annual-report green-investment compilation is separately withheld. It is not the new official CSMAR investment export. [Data access](docs/DATA_ACCESS.md) and input IDs D01-D06 distinguish the files and their uses.
- The paper's displayed aggregate statistics are distinct from reusable commercial-data extracts. They are reference snapshots, not raw data for rebuilding the estimates.
- Public availability of an official document does not by itself establish permission to redistribute its full text. The original PDFs are therefore linked, not bundled.

See [data access](docs/DATA_ACCESS.md). Missing disclosures remain missing; the package does not replace them with zero or supply synthetic data as empirical observations.

The public guide data include all 139 screened directions, 97 technical branches, 31 nominal technology families and unresolved coding decisions. The [field dictionary](docs/INPUT_SCHEMA.md#public-field-dictionary) explains roles, function codes, source pages and unknown values. These are conditional industry-domain classifications, not a company database or a yearly opportunity index.

## Start here

The original analytical environment used **Python 3.12.3**. Install the pinned requirements in a separate environment, then:

```bash
python run.py check
python run.py validate-inputs --panel /external/licensed/analysis_panel_submission_ready_v2_strict_calendar_gtos.csv
```

`check` uses no licensed data and runs no paper models. `validate-inputs` reads an externally supplied canonical panel, checks sample membership and stored GTOS, and reconstructs the guide design; it does not estimate coefficients or write results.

The local check excludes root-level `.git`, `.idea`, `venv` and `.venv` directories and Python cache directories. Other unlisted files still fail, including private data or local outputs covered by `.gitignore`. These exclusions do not authorize uploading the whole working directory: distribute only the files listed in the manifest.

The canonical panel contains commercial-derived data and is **not provided**. Its expected fields and preprocessing boundary are documented in [input schema](docs/INPUT_SCHEMA.md). Having access to a database is not enough to reconstruct all author-specific matching decisions; the precise input contract also matters.

## Explicit analysis commands

All empirical commands require a panel and a **new output file outside this directory**. Existing output files are never overwritten.

```bash
python run.py sample --panel /external/licensed/panel.csv --output /external/results/sample.json
python run.py core --panel /external/licensed/panel.csv --output /external/results/core.json
python run.py guide --panel /external/licensed/panel.csv --output /external/results/guide.json
python run.py rd --panel /external/licensed/panel.csv --rd /external/licensed/PT_LCRDSPENDING.xlsx --output /external/results/rd.json
```

Other commands are `history`, `investment`, `subsidy`, `projects`, `bootstrap`, `nb2`, `influence`, `shared`, `reassignment`, `tails`, `supplementary` and `descriptive`. Run `python run.py COMMAND --help` for inputs. `subsidy` and `projects` require restricted frozen classification ledgers in addition to the source export; they construct exposures, audit support and call the original same-sample comparison routine.

`supplementary` adds the original alternative controls, industry-code-source checks and earlier R&D/investment comparisons. `descriptive` reproduces the final Appendix A6/A7 calculations, including the original fixed tail caps of 16, 32 and 83. These are distinct from the generic within-subset percentile calculations returned by `reference_decomposition.build_decomposition_diagnostics`; use `descriptive` for the reported tail table.

`bootstrap` uses the original 2,000 firm draws and seed. `reassignment` uses the original 500 firm-label, 500 within-cell and 200 bundle reassignments. These are explicit opt-in commands, not part of `check`.

See [analysis map](docs/ANALYSIS_MAP.md) for the paper-to-code correspondence and remaining API-only diagnostics. There is no unattended `run all` command and no automatic Word editing, downloading, uploading or publication step.

## Empirical validation

On 13 October 2026, all 16 explicit empirical commands were run with the original inputs, samples, model settings and seeds, and outputs were compared with selected current reference results. Checks included all 2,000 component-bootstrap draws, 73 industry deletions, the shared-industry comparison, non-patent comparisons, and the original 500/500/200 reassignments. Final A6/A7 descriptive values were checked at their displayed precision.

Two numerical qualifications remain. Joint NB2 fits differ from the saved results by small floating-point amounts; coefficients, standard errors, p-values and intervals agree at four decimal places. The Poisson control-only model reproduces the original overflow and unavailable estimates under default BLAS threading, but converges under single-threaded BLAS with the same statistical specification. The reference table was not replaced with that additional estimate. Exact machine-level agreement is therefore not claimed.

The validation used Python 3.12.3 and the pinned package versions. Main runs used `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1`; the Poisson discrepancy was also checked in a separate process without those overrides. Internal inputs, run logs and detailed replication outputs are not bundled. This validation does not establish unrestricted third-party data access, measurement validity or causal identification.

Later maintenance added fixed-NB2 convergence checks and limited coefficient-path caching to one diagnostic call on an unchanged sample. Formulas, covariance estimators and dispersion calculations were retained. Selected deterministic outputs and summaries of the saved bootstrap draws were checked against the empirical validation record; this was not a rerun of all 16 commands.

The current 41-test suite covers numerical identities, calendar lags, missingness, convergence handling, cache isolation, public guide coding and release-file boundaries. It passes both in the local directory and in a temporary copy containing only the release files. These checks use synthetic fixtures and public classifications, not the paper's licensed inputs. They do not certify every conditional industry crosswalk or each firm's actual activity.

## Code provenance and limits

The retained analysis functions were copied from the current formal source code. Imports were made package-relative; project-specific mains, fixed local output paths and document-maintenance scripts were omitted. The final strict-calendar wrapper's exclusion of the descriptive `industry_name_std` field from common-sample requirements is retained.

Module aliases and console messages use descriptive names rather than internal revision-stage numbers. `provenance.json` records these readability changes. The AST checks reverse only those declared name and message substitutions before comparing functions with the original source hashes; numerical expressions and statistical rules must still match.

Some inherited helpers calculate additional columns that are not used in the final paper. Their presence is not evidence of an additional reported analysis. Numerical functions have not been refactored just to change writing style.

This is an analysis-input-level package. The complete raw-export-to-canonical-panel build and the restricted title ledgers are not distributed here. Full end-to-end empirical reproduction is therefore conditional on authorized access to the exact required inputs, and has not been claimed. [Release checklist](docs/RELEASE_CHECKLIST.md) records what still needs to be decided or verified before public release. Local `venv`, `.venv` and `.idea` directories are not release content.

## License and citation

No code or data license has been granted. Third-party rights are not changed. Article citation metadata, an archive DOI and licensing must be added only after they are confirmed; no provisional DOI or author information has been invented.

`release_manifest.json` lists the reviewed files and their SHA-256 hashes. It is not a certification of supplier permissions or empirical results. After approved edits, review and refresh the manifest before upload; do not treat `.gitignore` as a disclosure audit.
