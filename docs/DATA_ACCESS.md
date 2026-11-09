# Data access and redistribution boundary

The public candidate contains analysis code, author-created guide classifications and the paper's displayed aggregate results. It contains no company-level database records. The table below identifies the commercial inputs and other materials not included.

## Which data are restricted?

| Category | Data used in this study | Release treatment |
|---|---|---|
| Commercial provider inputs | CNRDS green invention patent records; CCAD agricultural-listed-firm membership (Zhejiang University-Qiyan); CSMAR financial, ownership, registered-location, industry, R&D, environmental-investment and green-subsidy inputs; CCER (Xenophon) environmental-penalty inputs | Not included. Obtain the required records through authorized provider access. |
| Macrodatas green investment compilation | D03: firm-level amounts compiled by Macrodatas from listed-company annual reports for 1999-2024. The authors match the records to their firm-year panel and construct the lagged industry measure for Table A9, Panel A. | Not included. This dataset is separate from CSMAR inputs D04-D06. |
| Restricted-source derived data | The canonical firm-year panel, matching and classification records, patent-based GTOS series, R&D/investment/subsidy industry-year series, and row-linked title classifications | Not included. |
| Author-created public-source classifications | Conditional technology-branch, industry and role coding of seven official 2017 R&D guides | Included in `data/public/`; no company records, monetary amounts or patent outcomes. Source documents are linked, not bundled. |
| Publication reference results | Tables already displayed in the manuscript and supplement | Included in `results/reference/paper_tables.json`. These are publication summaries, not reusable company or industry-year datasets. |

The [input schema](INPUT_SCHEMA.md) lists the file names, years, fields and uses for D01-D06. For D05-D06, CSMAR supplies the original project titles. The authors classify those titles and use the labels to calculate the industry shares; neither the original records nor the row-linked labels are included.

## Source documentation retained locally

On 14 October 2026, the author confirmed that the patent input comes from CNRDS, not Qiyan. The numerical input files and analysis are unchanged. Qiyan remains part of the CCAD agricultural-firm source name; the two datasets must not be conflated.

| Input | Local evidence | What this establishes |
|---|---|---|
| Green invention patent records | `01_raw_data/绿色发明专利/绿色专利研究数据库说明书.pdf`, the companion copyright/source notice, and the `绿色发明专利引用基本信息.xlsx` file read by the patent builder | CNRDS Green Patent Research Database; green coding follows the WIPO classification basis stated in its manual. The detailed applicant-to-listed-firm matching procedure is not supplied. |
| Agricultural-listed-firm membership | `01_raw_data/涉农上市公司数据库/中国涉农上市公司研究数据库字段说明.docx` | Required attribution: China Academy for Rural Development-Qiyan China Agri-research Database (CCAD), Zhejiang University. This is not the patent provider. |
| Registered location and listing information | `01_raw_data/326.上市公司注册地所在地数据（1991-2023年）/数据说明.txt` | The source note identifies CSMAR and listed-company annual reports. |
| Financial and ownership inputs | The financial field description and the financial/ownership workbooks' field-name and definition rows | Fields and definitions in the supplied exports. |
| R&D, environmental investment and green subsidies | CSMAR field descriptions and copyright notices in the R&D and new green-investment directories | Database identity, exported units and categories. |
| Environmental penalties | `01_raw_data/上市公司环保处罚2009-2024/数据来源说明.txt` and `上市公司环保处罚总表xlsx.xlsx` | CCER (Xenophon). Recorded announcements span 12 January 2009 to 30 April 2024; penalty regressions use 2009-2023, not the full patent-panel period. |
| Macrodatas green investment compilation, D03 | [Macrodatas dataset page](https://www.macrodatas.cn/article/1147473728), the local Stata dataset label and accompanying source note | Macrodatas compiled the dataset from listed-company annual reports and used the environmental-verification industry list. The source page identifies the compilation, not the authors' download route or redistribution rights. |

The [CNRDS access page](https://www.cnrds.com/Home/Login) provides registration and data-use-agreement procedures. Researchers need authorized access to the required provider modules. Source documentation identifies the datasets but does not establish account-specific access or redistribution permission, including for compilations and derived data. Restricted inputs are not offered on request.

## Reproduction boundary

The analysis functions require the canonical processed panel described in `INPUT_SCHEMA.md`. It is not generated from commercial sources by this release candidate. Readers must also reproduce or obtain the author-specific stock-code/year matching, industry priority, static CCAD grouping and missing-value decisions.

For the title-record analyses, access to CSMAR alone does not supply the author-reviewed frozen row labels. The package deliberately fails rather than silently replacing those labels with empty rules or a new classifier. The public guide comparison does not have this particular dependency: its author-created guide coding is bundled, while the firm panel remains restricted.
