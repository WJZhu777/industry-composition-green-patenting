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

`validate-inputs` checks the sample, stored GTOS variables and guide design without estimating models. Analysis outputs must be new files outside this repository. Some commands need additional R&D, investment or classification inputs; use `python run.py COMMAND --help` and the [analysis map](docs/ANALYSIS_MAP.md) for details. Some appendix diagnostics are available only through the Python API.

Before empirical runs, read the [reproduction notes](docs/RELEASE_CHECKLIST.md#reproduction-notes) for BLAS settings and known Poisson and NB2 numerical differences.

## Data

Commercial records from CNRDS (patents), CCAD (agricultural-firm membership), CSMAR (company attributes, R&D and environmental records) and CCER (penalties) are not included. The earlier annual-report green-investment compilation is also withheld.

The code starts from a prepared firm-year analysis panel, which is not distributed. It does not recreate that panel from raw database exports. Reproduction requires authorized access to the exact inputs and matching rules described in the [input schema](docs/INPUT_SCHEMA.md); a database subscription alone does not supply the author-specific panel or frozen project-title labels. See [data access](docs/DATA_ACCESS.md) for sources and restrictions.

`data/public/` contains author-created classifications of seven official 2017 R&D guides and links to their sources. Original PDFs are not bundled. `results/reference/` contains the paper's displayed tables for comparison, not input data for re-estimation.

## Documentation

- [Analysis map](docs/ANALYSIS_MAP.md): commands, paper-to-code links and display-label differences.
- [Input schema](docs/INPUT_SCHEMA.md): required fields and the public classification dictionary.
- [Data access](docs/DATA_ACCESS.md): commercial sources and redistribution boundaries.
- [Release checklist](docs/RELEASE_CHECKLIST.md): reproduction notes and remaining publication decisions.

`provenance.json` records the retained functions and source checksums. `release_manifest.json` lists the reviewed files and their hashes.

The repository is currently private. No public license has been selected, and no Zenodo archive has been created.
