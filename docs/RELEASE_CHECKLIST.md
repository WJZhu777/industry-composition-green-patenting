# Before public upload

## Decisions still required

- Choose code and author-created-data licenses, if desired. No license has been added at the author's request.
- Confirm which third-party access and redistribution terms apply. Do not upload restricted panels, exports or row labels.
- Review the input boundary already documented in `DATA_ACCESS.md` and `INPUT_SCHEMA.md`: commercial exports, the earlier disclosure compilation, their derived panel/series and row-linked labels are not bundled. Do not promise delivery without an authorized route.
- Decide whether to add the remaining raw preprocessing. The current analysis-input-level reproduction boundary is disclosed; supplementary input adapters do not supply the unavailable panel/ledgers.
- Confirm final author/citation information, accepted article title and publication DOI before adding citation metadata.

## Verification

- Completed empirical checks and numerical qualifications are recorded in [Reproduction notes](#reproduction-notes).
- Review the remaining API-only appendix coverage and the raw-input reproduction boundary; successful explicit commands do not certify every historical helper or supply restricted inputs.
- Review the final allowlist and hashes after any new edits; `.gitignore` is only a convenience and does not protect already tracked files.
- Keep the public guide codebook and pending reviews with the classifications. Source traceability and matrix checks do not establish universal firm eligibility or certify unresolved industry subtypes.
- Confirm that no new local outputs, credentials, restricted exports, raw PDFs, review correspondence or document versions have entered the directory.

## GitHub distribution

The code and author-created public-source classifications are hosted in the [GitHub repository](https://github.com/WJZhu777/industry-composition-green-patenting). As part of submitting the revised manuscript, make the repository public and verify access without signing in. Only the reviewed manifest files belong in the repository; restricted inputs are excluded. No public license has been selected.

Keep generated empirical outputs outside this folder. Only deliberately reviewed aggregate publication reference results should be considered for a future release.

## Reproduction notes

On 13 October 2026, all 16 explicit empirical commands were run with the original inputs, samples, specifications and seeds. Selected current outputs were compared, including all 2,000 component-bootstrap draws, 73 industry deletions, the shared-industry comparison and the 500/500/200 reassignments. Final A6/A7 descriptive values were checked at their displayed precision. Internal inputs, logs and detailed replication outputs are not bundled.

The runs used Python 3.12.3 and the pinned dependencies, with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set before starting Python. Two numerical qualifications remain:

- Joint NB2 fits differ from the saved results by small floating-point amounts. Coefficients, standard errors, p-values and intervals agree at four decimal places.
- The Poisson control-only model reproduces the original overflow and unavailable estimates under default BLAS threading, but converges under single-threaded BLAS with the same specification. That additional estimate did not replace the reference table.

The Poisson output keeps the raw `optimizer_converged` flag. Use `fit_usable` to select valid results: it requires convergence and finite parameters, standard errors, p-values, confidence intervals and log likelihood. `converged` also reflects this check. Unusable estimates are returned as missing. The code does not retry with a different optimizer or model.

Later maintenance added fixed-NB2 convergence checks and confined coefficient-path caching to one call on an unchanged sample. Selected deterministic outputs and saved-bootstrap summaries were checked; this was not another run of all 16 commands. The synthetic/public-data test suite also passed in the local package and in a copy containing only release files. These checks are distinct from empirical reproduction with licensed inputs.

On 28 October 2026, `sample-bridge` and `wild-cluster` were added for Appendix Table A3 Panel G and Table A2 Panel N. Their numerical functions retain the checked analytical sources. The local package and an isolated release-file copy each passed 54 synthetic/public-data tests, including fixed sample selection, test order, seeded weights and agreement with direct bootstrap refits. Read-only checks against the licensed inputs matched all 14 empirical sample keys and design-column lists, all 28 random-weight arrays, and the three sample-bridge designs. The portable empirical commands were not rerun at that stage: the existing results and manuscript estimates were not replaced. Display snapshots include both new panels. These changes were included in the 7 November push at commit `51e7c1b`.

On 8 November 2026, the contextual penalty models were restricted to 2009-2023, excluding the uncovered early years and incomplete 2024. The new `penalties` command re-estimated the 12 Table A5 specifications. All 47,736 covered panel indicators matched the raw announcement records. Independent matrix calculations matched the coefficients and firm-clustered inference within 2e-13. Panels B-C use 42,332 observations; the patent panel and its 47,773-row common sample are unchanged. The simplest Panel A coefficient is significant at 10%, not 5%; the later-period conditional difference remains positive.

The updated package passed 65 synthetic/public-data tests and imported all 36 modules. Input validation also confirmed the original sample flow and strict-calendar GTOS construction without refitting patent models.

`provenance.json` records the original functions and the declared import-alias and console-message changes. Function checks reverse those substitutions before comparing AST hashes. Separate release hashes record validation changes and the penalty-only coverage restriction in the legacy baseline API. The strict-calendar sample rule still excludes the descriptive `industry_name_std` field from common-sample requirements. Some inherited helpers return additional columns not reported in the paper; document-generation and historical exploratory scripts are not included.
