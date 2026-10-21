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

## Private GitHub repository

The author approved a private repository at [WJZhu777/industry-composition-green-patenting](https://github.com/WJZhu777/industry-composition-green-patenting) on 20 October 2026. Only the reviewed manifest files belong in that repository. Public release requires a separate author decision. No Zenodo record, reserved DOI or public license has been created.

Keep generated empirical outputs outside this folder. Only deliberately reviewed aggregate publication reference results should be considered for a future release.

## Reproduction notes

On 13 October 2026, all 16 explicit empirical commands were run with the original inputs, samples, specifications and seeds. Selected current outputs were compared, including all 2,000 component-bootstrap draws, 73 industry deletions, the shared-industry comparison and the 500/500/200 reassignments. Final A6/A7 descriptive values were checked at their displayed precision. Internal inputs, logs and detailed replication outputs are not bundled.

The runs used Python 3.12.3 and the pinned dependencies, with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set before starting Python. Two numerical qualifications remain:

- Joint NB2 fits differ from the saved results by small floating-point amounts. Coefficients, standard errors, p-values and intervals agree at four decimal places.
- The Poisson control-only model reproduces the original overflow and unavailable estimates under default BLAS threading, but converges under single-threaded BLAS with the same specification. That additional estimate did not replace the reference table.

Later maintenance added fixed-NB2 convergence checks and confined coefficient-path caching to one call on an unchanged sample. Selected deterministic outputs and saved-bootstrap summaries were checked; this was not another run of all 16 commands. The synthetic/public-data test suite also passed in the local package and in a copy containing only release files. These checks are distinct from empirical reproduction with licensed inputs.

`provenance.json` records the original functions and the declared import-alias and console-message changes. Function checks reverse only those declared substitutions before comparing AST hashes. The strict-calendar sample rule still excludes the descriptive `industry_name_std` field from common-sample requirements. Some inherited helpers return additional columns not reported in the paper; document-generation and historical exploratory scripts are not included.
