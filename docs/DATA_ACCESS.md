# Data access and redistribution boundary

This repository contains analysis code, author-created guide classifications and the aggregate results shown in the manuscript and supplement. It does not contain company-level database records.

## Which data are restricted?

| Category | Data used in this study | Release treatment |
|---|---|---|
| Restricted database inputs | CNRDS green invention patent records; CCAD agricultural-listed-firm membership; CSMAR financial statements, primary industry codes, R&D, environmental-investment and green-subsidy records; CCER (Xenophon) environmental-penalty records; Macrodatas firm-level green investment amounts | Not included. Request access from the relevant provider. |
| Other inputs to the firm-year panel | The Listed Company Green Credit Data (2007-2024) compilation supplies the firm-year list and fallback industry codes. The analysis does not use its credit measures. An annual ownership table supplies state ownership. A company register compiled from CSMAR and annual reports supplies listing years and registered locations. | Not included. |
| Restricted-source derived data | The canonical firm-year panel, matching and classification records, patent-based GTOS series, R&D/investment/subsidy industry-year series, and row-linked title classifications | Not included. |
| Author-created public-source classifications | Conditional technology-branch, industry and role coding of seven official 2017 R&D guides | Included in `data/public/`; no company records, monetary amounts or patent outcomes. Source documents are linked, not bundled. |
| Reference tables | Aggregate tables reported in the manuscript and supplement | Included in `results/reference/paper_tables.json`. The manuscript and supplement files are not included. |

The [input schema](INPUT_SCHEMA.md) lists the file names, years, fields and uses for D01-D06. For D05-D06, CSMAR supplies the original project titles. The authors classify those titles and use the labels to calculate the industry shares; neither the original records nor the row-linked labels are included.

## Data sources and access

Green invention patent records come from the [CNRDS Green Patent Research Database](https://www.cnrds.com/Home/Login). CNRDS uses the WIPO green patent classification. The provider documentation does not specify the detailed procedure used to match applicants to listed firms.

Agricultural-listed-firm membership comes from the China Academy for Rural Development-Qiyan China Agri-research Database (CCAD) at Zhejiang University. For Table A9, Panel A, the authors match [green investment amounts compiled by Macrodatas](https://www.macrodatas.cn/article/1147473728) from listed-company annual reports to the firm-year panel and construct the lagged industry measure. The [input schema](INPUT_SCHEMA.md) identifies the other provider tables and their analytical coverage.

Request access to restricted records from the relevant providers under their terms. The authors do not distribute restricted inputs or derived firm-level data.

## Reproduction boundary

The analysis code starts from the prepared firm-year panel described in `INPUT_SCHEMA.md`. This repository does not rebuild that panel from raw files. To construct it, readers need the source files and the study's rules for matching firms and years, selecting industry codes, assigning CCAD status and handling missing values.

The title-record analyses require both the CSMAR exports and the authors' row-linked classifications. The guide classifications are included, but the guide regressions still require the restricted firm-year panel.
