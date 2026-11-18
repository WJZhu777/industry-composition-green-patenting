# Industry composition and green invention patenting

Replication code for **Industry Composition and the Green Invention Patenting Gap between Agricultural and Other Listed Firms in China**. The study compares patent applications across and within industries and uses descriptive decompositions to examine the overall gap.

## Run

The analysis used Python 3.12.3. From the repository root, in a virtual environment:

```bash
python -m pip install -r requirements.txt
python run.py check
python run.py --help
```

`check` runs synthetic and public-data tests. It does not need licensed data.

To run the main analysis, supply the prepared firm-year panel. Replace the example paths with your own:

```bash
python run.py validate-inputs --panel ../licensed/panel.csv
python run.py core --panel ../licensed/panel.csv --output ../results/core.json
```

Use a new output file outside the repository. See the [reproduction notes](docs/RELEASE_CHECKLIST.md) for BLAS settings and the [analysis map](docs/ANALYSIS_MAP.md) for commands and table references.

## Data

The repository does not include the firm-year panel, licensed data exports or project-title labels linked to those exports. It starts from the prepared panel and does not rebuild it from raw data. See [data access](docs/DATA_ACCESS.md) for the providers and the [input schema](docs/INPUT_SCHEMA.md) for the required files and fields.

`data/public/` contains the study's classifications of seven official 2017 R&D guides and links to the source documents. `results/reference/` contains the reported tables, not data for re-estimation. The manuscript and source PDFs are not included.
