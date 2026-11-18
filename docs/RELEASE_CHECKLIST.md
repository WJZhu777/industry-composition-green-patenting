# Reproduction notes

This repository provides analysis code and guide classifications. It does not include the prepared firm-year panel, restricted source files or row-linked project classifications. See [Data access](DATA_ACCESS.md) and the [input schema](INPUT_SCHEMA.md) for details. No license is provided in this repository.

## Empirical checks

Replication checks with the original inputs covered the main models and selected supplementary results, including the 2,000 component-bootstrap draws, industry deletions, the shared-industry comparison and the reassignments. Tables A6 and A7 were checked at the reported precision.

The runs used Python 3.12.3 and the pinned dependencies, with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OMP_NUM_THREADS=1` set before starting Python. Two numerical qualifications remain:

- Joint NB2 fits differ from the saved results by small floating-point amounts. Coefficients, standard errors, p-values and intervals agree at four decimal places.
- The Poisson control-only model reproduces the original overflow and unavailable estimates under default BLAS threading, but converges under single-threaded BLAS with the same specification. That additional estimate did not replace the reference table.

The Poisson output keeps the raw `optimizer_converged` flag. Use `fit_usable` to select valid results: it requires convergence and finite parameters, standard errors, p-values, confidence intervals and log likelihood. `converged` also reflects this check. Unusable estimates are returned as missing. The code does not retry with a different optimizer or model.

Run `python run.py check` to test the package with synthetic fixtures and bundled public data. This does not re-estimate the paper's models.

`sample-bridge` and `wild-cluster` correspond to Appendix Table A3 Panel G and Table A2 Panel N. Verification matched the samples, design matrices and wild-bootstrap weights without re-estimating those results.

`penalties` reproduces the 12 specifications in Table A5. The code uses 2009-2023 because the penalty records begin in 2009 and cover only part of 2024. The penalty indicators were checked against the source announcements, and independent matrix calculations reproduced the coefficients and firm-clustered inference.
