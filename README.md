# Industry composition and green invention patenting

Research code and public-source classifications accompanying **Industry Composition and the Green Invention Patenting Gap between Agricultural and Other Listed Firms in China**, submitted to *Environment, Development and Sustainability*.

The study examines how industry composition relates to differences in green invention patent applications between agricultural and other Chinese listed firms. The analyses are descriptive: industry-year comparisons, accounting decompositions and supplementary models.

## Contents

```text
run.py                  Command-line entrypoint
requirements.txt        Pinned dependencies
code/research_release/  Analysis functions
data/public/            Guide classifications and source links
results/reference/      Tables displayed in the paper
tests/                  Synthetic and public-data checks
docs/                   Data requirements and analysis instructions
```

## Run

The original analysis used Python 3.12.3. Use a separate virtual environment and run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py --help
```

`check` tests numerical identities, calendar lags, missing-value rules and public classifications using synthetic fixtures and bundled public data. It does not run the paper's empirical models or require commercial records.

With the required analysis panel, first validate the input, then run a selected analysis. Replace the example paths with your own:

```bash
python run.py validate-inputs --panel ../licensed/panel.csv
python run.py core --panel ../licensed/panel.csv --output ../results/core.json
```

`validate-inputs` checks the fields required by `core`, the 2015 split, the sample, stored GTOS variables and the guide design without estimating models. Analysis outputs must be new files outside this repository. The final appendix diagnostics use `sample-bridge` and `wild-cluster`. Some commands need additional R&D, investment or classification inputs; use `python run.py COMMAND --help` and the [analysis map](docs/ANALYSIS_MAP.md) for details. Some appendix diagnostics are available only through the Python API.

Before empirical runs, read the [reproduction notes](docs/RELEASE_CHECKLIST.md#reproduction-notes) for BLAS settings and known Poisson and NB2 numerical differences.

Use `penalties` for Appendix Table A5. It restricts the contextual penalty comparison to 2009-2023, without changing the patent samples.

## Data

The repository does not include restricted records from CNRDS, CCAD, CSMAR, CCER or Macrodatas.

The code starts from a prepared firm-year panel. The panel is not included because it incorporates commercial database records. The code does not rebuild it from raw exports. The [input schema](docs/INPUT_SCHEMA.md) lists the files and fields needed to run the analyses, including the authors' project-title labels linked to CSMAR records. See [data access](docs/DATA_ACCESS.md) for sources and what the repository provides.

`data/public/` contains author-created classifications of seven official 2017 R&D guides and links to their sources. Original PDFs are not bundled. `results/reference/` contains the paper's displayed tables for comparison, not input data for re-estimation.

## Documentation

- [Analysis map](docs/ANALYSIS_MAP.md): commands, paper-to-code links and display-label differences.
- [Input schema](docs/INPUT_SCHEMA.md): required fields and the public classification dictionary.
- [Data access](docs/DATA_ACCESS.md): commercial sources and redistribution boundaries.
- [Reproduction notes](docs/RELEASE_CHECKLIST.md): checks performed and numerical differences to expect.

`provenance.json` records checksums for the retained analysis functions. `release_manifest.json` lists the files included in this repository and their hashes.
