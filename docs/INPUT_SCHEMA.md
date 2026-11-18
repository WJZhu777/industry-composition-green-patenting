# Required analysis inputs

## Data inventory

Source coverage describes the export or document edition. Outcome years describe the observations used in the analysis, which ends in 2024.

### Empirical input files

| ID | Dataset and source | Required file | Source coverage | Actual use / outcome years | Fields and units used | Paper location / command | Included in this package? |
|---|---|---|---|---|---|---|---|
| D01 | Prepared firm-year panel. The sources used to build it are described below. | `analysis_panel_submission_ready_v2_strict_calendar_gtos.csv` | Processed panel: 2007-2024 | Full-panel comparisons: 2007-2024; common estimation sample: 2008-2024; FI baseline: 2007-2012 | Application counts; binary entry, penalty and CCAD flags; financial controls; year/province/industry categories; OI in natural-log units, OD as a fraction, FI as 0/1. Detailed fields are listed below. | Main Tables 1-6; A1-A8; matching backbone for A9; `sample`, `core`, `bootstrap`, `nb2`, `influence`, `shared`, `reassignment`, `tails`, `descriptive` | No: panel derived from restricted inputs |
| D02 | CSMAR listed-company R&D input table | `PT_LCRDSPENDING.xlsx` | Raw records: 2000-2024; selected regular consolidated annual records: 2007-2024 | Lagged R&D comparisons: 2008-2024; personnel supplement: 2016-2024; own-history intersection: 2008-2024. Each exercise retains its own fixed eligible sample. | `RDSpendSum`: yuan as exported; `RDSpendSumRatio`, `RDPersonRatio`: reported percentages; `RDPerson`: persons. P2 ratio regressions use percentage points. The earlier monetary environment is log(1 + sum of disclosed R&D / number of all non-agricultural firms in the industry-year), not a mean restricted to disclosers. | A9 Panels A-D; A2 alternative-control and own-history comparisons; `rd`, `history`, `supplementary` | No: commercial export |
| D03 | Macrodatas compilation of listed-company annual-report disclosures, scoped using the environmental-verification industry list | `上市公司绿色投资数据（1999-2024年）.dta` | 1999-2024 | Matched source records restricted to 2007-2024; supported time-adjusted capitalized-investment cells: 2009-2024 | `股票代码`, `年份`, `绿色总投资`, `资本化绿色投资额`, `费用化绿色投资额`, `绿色投资2`. The authors match the records by firm and year and calculate the lagged industry mean of log(1 + capitalized amounts) in source monetary units. The Stata label identifies the capitalized field as summed period-end carrying value, not annual additions. The helper also retains the source capitalized/total fraction (`绿色投资2`) and total/expense amounts. | A9 Panel A capitalized investment comparisons; `supplementary` | No: Macrodatas source records are not included |
| D04 | CSMAR Overall Environmental Investment Table | `GI_EnvInvestTotal.xlsx` | 2000-2025 | Lagged expense and capital ratio comparisons: 2008-2024, in separate fixed samples | `ExpenseToGrossRevenue`, `CapitalToTotalAssets`: exported percentage-point ratios. Monetary component fields are in yuan but are not summed to replace these reported ratios. | A9 Panels E-F; `investment` | No: commercial export |
| D05 | CSMAR Detailed Environmental Investment Table | `GI_EnvInvestDetail.xlsx` | 2000-2025 | Source-code 5 title scope: 2007-2024, with matched outcomes 2009-2024; source-code 2 title scope: 2018-2024, with matched outcomes 2019-2024 | `Symbol`, `EndDate`, `InvestSourceCode`, `Item`, source-row ID. Source 2 is research expense; source 5 is development expenditure. The model inputs are title-record shares in percentage points, not monetary investment. `Amount` is exported in yuan and was inspected for accounting/availability diagnostics, not used to form the reported P20 exposure. | A9 Panels G1-G3, source-specific project-record comparisons; `projects` | No: commercial export |
| D06 | CSMAR Listed-Company Government Green Subsidy Information Table | `GI_LstdGovGrnSubLetter.xlsx` | 2003-2025 | Consolidated source records restricted to 2007-2024; lagged title-share outcomes: 2008-2024 | `Symbol`, `EndDate`, `StateTypeCode`, `Item`, source-row ID. The exposure is the percentage-point share of recorded peers with an explicitly green R&D title. `Amount` is exported in yuan but is not the regression exposure or an annual R&D funding total. | A9 Panels G1-G3, subsidy-title comparison; `subsidy` | No: commercial export |

For D01, the Listed Company Green Credit Data (2007-2024) compilation (`上市公司绿色信贷数据（2007-2024）.xlsx`) supplies the firm-year list and fallback industry field. The analysis does not use its credit measures. CSMAR supplies the primary industry field; the 2023 and 2024 codes use the fallback. CNRDS supplies patent records, CCAD supplies agricultural-firm membership, and CSMAR supplies financial statements. State ownership comes from an annual ownership table. Listing years and registered locations come from a company register compiled from CSMAR and annual reports (1991-2023). See Appendix Table A1.

For the D01 penalty indicator, Table A5 and the `penalties` command use 2009-2023. The CCER export contains announcements from 12 January 2009 to 30 April 2024. Within the analysis window, zero means no matched announcement, not proof of no violation. This restriction applies only to the penalty models.

Use the paths to your own inputs when running the commands. Keep CSMAR's two metadata rows after the header when preparing the Excel exports.

### Frozen classification inputs

| Input file | Contents and scope | Required by | Included? |
|---|---|---|---|
| `p17_green_rd_subsidy_feasibility.json` | Frozen title-purpose categories linked to source rows and literal titles; selected consolidated records from 2007-2024. D06 and the frozen labels must agree. | `subsidy --labels`; A9 subsidy comparison | No: external restricted classification ledger |
| `p19_green_rd_project_feasibility.json` | `analysis.title_ledger` for the selected source-code 2/5 investment records; recorded categories, row IDs, accounting-adjustment flags and source-specific scope. D05 and the frozen labels must agree. | `projects --labels`; A9 project-record comparisons | No: external restricted classification ledger |

The database exports do not include these classifications. Reproducing the reported title-share comparisons requires the original labels matched to the source rows, not a new keyword classification. The labels contain no monetary amounts.

### Bundled guide data and underlying documents

| Bundled file | Contents | Edition / analytical use | Units |
|---|---|---|---|
| `data/public/guide_coding.json` | Frozen author-created conditional industry/role coding; complete 139-direction screening record, 97 branch records, 31 technology-family labels, industry-role matrix and pending reviews | Seven 2017 guides and the 2011 industry standard; A9 Panels H1-H2 / `guide`, 2018-2020 role-separated samples | Categorical labels and binary function flags; no annual opportunity count or monetary measure |
| `data/public/guide_branches.csv` | Readable branch-level view of the same coding, including source IDs, pages, conditional codes and scope | Same frozen coding; not a separate dataset or additional observations | Categorical/text fields |
| `data/public/guide_industry_roles.csv` | Readable industry/role/family view; retains unknown and unlinked roles | Same frozen coding; not verified firm-level R&D eligibility | Categories and function flags |
| `data/public/official_sources.json` | Original source IDs, official URLs and frozen PDF SHA-256 values | Seven 2017 guides plus GB/T 4754-2011; provenance for the guide comparison | Document metadata, not firm-year observations |

The seven source IDs refer to the following official project-application guide editions. Document names below are transcribed from the preserved source headings; the links and fingerprints are in `official_sources.json`.

| Source ID / standard | Source document | Edition |
|---|---|---|
| `agro_2017` | "化学肥料和农药减施增效综合技术研发"试点专项项目申报指南 | 2017 |
| `air_2017` | "大气污染成因与控制技术研究"试点专项项目申报指南 | 2017 |
| `water_2017` | "水资源高效开发利用"重点专项项目申报指南 | 2017 |
| `soil_2017` | "农业面源和重金属污染农田综合防治与修复技术研发"重点专项项目申报指南 | 2017 |
| `food_2017` | "现代食品加工及粮食收储运技术与装备"重点专项项目申报指南 | 2017 |
| `livestock_2017` | "畜禽重大疫病防控与高效安全养殖综合技术研发"重点专项项目申报指南 | 2017 |
| `forest_2017` | "林业资源培育及高效利用技术创新"重点专项项目申报指南 | 2017 |
| `GB/T 4754-2011` | 国民经济行业分类; activity definitions used in conditional links to detailed CSRC 2012 codes | 2011 standard, not an annual industry assignment |

The repository includes the author-created coding and source metadata, not the original guide or standard PDFs. A 2017 guide edition is not evidence of a continuously observed opportunity series for 2007-2024. Earlier feasibility routes not reported in the manuscript, including the technology-catalogue and public-laboratory screens, are not inputs to the final guide comparison.

### How the public classifications were constructed

All 139 directions in the seven selected guides were screened. Seventy-five contain retained technical branches, 59 are outside the narrow technical scope, and five remain unresolved at direction level. The 97 branch records split mixed tasks where necessary; they are not 97 distinct inventions or equal-sized opportunities. The 31 technology families are nominal subject-matter labels, not scores. This is a complete screening ledger for these seven documents, not a census of all national R&D programs.

A named new product, equipment item or software supports a conditional `supplier` link. A named new operating or manufacturing method supports a conditional `process_rd` link. Adoption, raw-material supply or a demonstration site alone supports neither development role. Possible technology use is recorded separately as `user`; that role is not used in the reported guide regressions. Detailed CSRC links use the activity definitions of GB/T 4754-2011, with product/activity conditions retained in each branch. They do not establish that every firm in an industry carries out that activity or could apply for the project.

The coding was revised after an earlier comparison had been observed and was frozen before the final role-separated comparison. It was not blinded or preregistered. The recorded corrections use technical content and industry-activity definitions; they do not select branches using coefficient values or significance. Retained provisional links and nine pending reviews are disclosed, including unresolved manufacturer subtypes. No unresolved role is filled with zero.

### Public field dictionary

| Field / object | Meaning and reading rule |
|---|---|
| `direction_screening` | One record per source direction: original title and section, source ID, one-based PDF page, screening status, review reason and retained branch IDs. A direction may contain several branches, including a branch without a resolved commercial-industry role. |
| `branch_id`, `direction_id`, `source_id` | Stable identifiers. The prefix identifies the guide; the direction ID identifies its numbered task; a branch suffix identifies a coded part of that task. IDs are not company IDs. |
| `direction_title`, `literal_source_anchors` | Source heading and short search terms for locating the branch in the linked document. They do not reproduce the full task or replace its conditions. |
| `page_start_1based` | Page in the frozen PDF, counting its first page as 1; not a journal page or an annual exposure date. |
| `conditional_role_codes` | Separate lists of detailed CSRC codes for `supplier`, `process_rd` and `user`. A code is linked only under the branch's stated activity conditions. |
| `standard_activity_codes` | Four-digit GB/T activity references where recorded during targeted review. An empty list means none was recorded here, not that the CSRC link lacks an industry definition. |
| `scope_and_role_conditions`, `review_status`, `pending_reviews` | Technical restrictions and unresolved decisions. A provisional status is not independent expert certification. |
| `technology_family`, `technology_families` | Nominal branch family and its source membership. Family breadth differs and categories can overlap; counts are not opportunity intensity. |
| `industry_role_technology_matrix` | One record per linked industry in the union of all three roles, with a separate role object. Source branch IDs retain the lineage of every function and family flag. |
| `conditional_documented_domain` | At least one branch links this industry to this role under stated conditions. A function flag of 1 means a linked branch carries that function; 0 means none in this fixed ledger. |
| `unknown_or_unlinked_role` | No resolved link for that role in this ledger. Function flags are JSON `null` and empty CSV cells, not six zeros. Unlisted industries are likewise not interpreted as zero opportunity. |
| `official_sources.json` | Official document title, edition year, URL, PDF page count and SHA-256 of the frozen bytes. Edition year is not necessarily the web publication year. |

Function codes are: `P`, pollution/emissions or environmentally harmful input reduction; `W`, water saving/reuse/nonconventional water production; `E`, energy saving/recovery; `R`, material/feed/food loss reduction or residue reuse; `K`, ecosystem conservation/restoration or soil-maintaining forestry; `M`, environmental/resource monitoring or technical forecasting. These definitions are also bundled in `guide_coding.json`.

For each industry and development role, the code collects the complete set of `(technology_family, function)` pairs from the linked branches. Industries with exactly the same set receive the same categorical profile. Duplicate evidence does not increase a score; profile identifiers have no numerical order. The guide model uses profile indicators, omitting one reference profile, rather than adding branch counts or treating six functions as interchangeable with the full profiles. The fixed 2018-2020 samples are constructed separately for the two development roles; no annual guide series is generated.

The two CSV files are readable projections of the JSON, not additional observations. List/dictionary cells use JSON text. `guide_coding.json` retains the full source lineage and unresolved decisions; use it for the analysis and audit rather than interpreting CSV blanks as observed zeros.

This inventory documents the reported input boundary. It does not supply the commercial raw-export-to-canonical-panel build or promise access to restricted labels.

The final `sample-bridge` diagnostic needs only D01. The 14-test `wild-cluster` command needs D01, D02 and the bundled guide coding: these reconstruct the original common, positive-count, own-history and guide samples. It does not require internal result files or private firm-key/profile ledgers. Neither command changes the original GTOS construction.

## Canonical panel

One row per stock code and calendar year; UTF-8 CSV. Stock codes are six-character strings with leading zeros. The selected paper panel covers 2007-2024 and has 55,588 rows / 4,582 firms. The original stage counts are 55,588, 55,588, 51,696, 49,982, 48,519 and 47,773; the common sample has 4,290 firms. These are validation anchors, not instructions to force a different dataset into the same counts.

| Columns | Definition / contract |
|---|---|
| `证券代码`, `year` | Unique firm-year keys |
| `是否涉农` | Static CCAD listed-firm status, binary; not time-varying sector A status |
| `green_patent_application_count` | Nonnegative integer green invention application count |
| `log1p_green_patent_application_count` | Natural log of one plus count |
| `has_green_patent_application` | One if count is positive |
| `is_penalized_announcement_year` | Stored penalty indicator; retained in original common-sample membership |
| `ln_assets`, `roa`, `leverage`, `ocf_ratio`, `soe`, `list_age` | Original six controls with the original units and definitions |
| `province` | Registered province category, not a firm fixed effect |
| `industry_code_std` | Standardized detailed CSRC code; unavailable classifications retained as `UNKNOWN` in the loader |
| `industry_year_nonagri_patent_per_firm_lag1_log1p` | OI: log(1 + non-agricultural patent applications per firm) in the exact preceding calendar year |
| `industry_year_has_patent_share_nonagri_lag1` | OD: preceding-calendar-year non-agricultural patent-entry share |
| `frontier_industry_flag_baseline` | FI: fixed 2007-2012 high-intensity industry classification; name retained for compatibility, not a persistent frontier claim |

`core` also requires `industry_year_patent_count_nonagri`, `industry_year_patent_firms_nonagri`, `firm_count_nonagri`, and `baseline_nonagri_patent_per_firm`. Both `validate-inputs` and `core` list missing fields before estimation. The code constructs `post_2015` as `year >= 2015`; a supplied column must match that rule on every row. It is not a sample filter.

The v2 strict-calendar panel, rather than the legacy v1 panel, is the input. A non-agricultural industry-year with firms and no applications has a zero benchmark; an unavailable preceding year remains missing. Agricultural applications are excluded from the benchmark. `industry_name_std` is descriptive and not required for common-sample membership.

The inherited leave-one-out helper uses the previous observed firm row for own-history subtraction. This is equivalent to the preceding year in the selected canonical panel, whose firm histories were checked as consecutive; a different panel with gaps must be reviewed rather than accepted silently. The P8 history helper independently uses exact calendar-year joins.

## CSMAR R&D export

`PT_LCRDSPENDING.xlsx`, original two metadata rows after the header. Fields: `Symbol`, `EndDate`, `Source`, `StateTypeCode`, `Currency`, `RDPerson`, `RDPersonRatio`, `RDSpendSum`, `RDSpendSumRatio`, `RDExpenses`, `RDInvest`, `RDInvestRatio`, `RDInvestNetprofitRatio`.

Keep year-end annual consolidated records (`Source == 0`, `StateTypeCode == 1`, December 31), unique firm-year keys, original reported percentage units, and missing values. At least three non-agricultural disclosures are needed for the industry-year R&D environment. Personnel comparison starts in 2016. General R&D is not green-specific.

## CSMAR environmental-investment and subsidy exports

The original Excel export readers preserve metadata-row handling and source-row IDs. Inputs used in the paper include `GI_EnvInvestTotal.xlsx`, `GI_EnvInvestDetail.xlsx` and `GI_LstdGovGrnSubLetter.xlsx`. Required field names are checked by the reader functions; do not rename amount, ratio or source-code fields by intuition.

- Overall investment: `ExpenseToGrossRevenue` and `CapitalToTotalAssets`, as reported. Missing is not zero; a reported zero is preserved.
- Detail records: `Symbol`, `EndDate`, `InvestSourceCode`, `Item`, and amount-related fields used by the original reader. Source 2 and source 5 are analyzed separately, not summed into annual investment.
- Subsidy records: consolidated annual scope, original `Item` titles and source rows. Record availability defines the denominator; no-record firms are not classified as nonrecipients.
- Frozen classification ledgers: the internal P17/P19 JSON structures containing `title_ledger`, source rows and categories. These are external restricted inputs, not bundled release data. Altering an export's row order invalidates row-linked labels.

The `subsidy` and `projects` commands only accept a ledger that matches its export. Exposures use the original previous-calendar-year recorded non-agricultural peers and at least three remaining reporters after focal-firm exclusion.

## Supplementary inputs

`supplementary` requires the full canonical panel, the R&D export above, and the original 1999-2024 Stata green-investment dataset. That dataset must supply `股票代码`, `年份`, `绿色总投资`, `资本化绿色投资额`, `费用化绿色投资额` and `绿色投资2` in their original units. Its reader constructs stock-code/year keys and uses canonical-panel agricultural and industry assignments; it does not substitute group labels from another source. Earlier green-investment correlations and the new CSMAR ratio regressions use different inputs and must not be conflated.

The full panel must also retain `industry_name_std`, `industry_code`, `industry_code_std_source`, `ln_revenue`, `revenue_used_y`, `staff_count` and `penalty_count_announcement_year` for the supplementary routines. These fields cannot be recovered from a minimal panel. Alternative R&D controls use `RDSpendSumRatio / 100` as a fraction, with the original 1%/99% caps and exact previous-calendar-year matching; this differs from the uncapped percentage-point environment in P2. Earlier construct-comparison helpers also retain ratio-fraction diagnostics. Source sensitivity reconstructs GTOS separately under the original source rules.

Some diagnostics remain API-only, as listed in `ANALYSIS_MAP.md`.
