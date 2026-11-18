# Data access

The study uses the sources below. Request restricted records directly from the providers under their terms.

| Source | Used for |
|---|---|
| [CNRDS Green Patent Research Database](https://www.cnrds.com/Home/Login) | Green invention patent records. CNRDS uses the WIPO green patent classification. Its documentation does not describe the full applicant-to-listed-firm matching procedure. |
| China Academy for Rural Development-Qiyan China Agri-research Database (CCAD), Zhejiang University | Agricultural-listed-firm membership |
| CSMAR | Financial statements, primary industry codes, R&D, environmental investment and green subsidies |
| CCER (Xenophon) | Environmental-penalty announcements |
| [Macrodatas green investment compilation](https://www.macrodatas.cn/article/1147473728) | Amounts compiled from listed-company annual reports. The authors match the amounts to the firm-year panel and construct the lagged industry measure for Table A9, Panel A. |

The [input schema](INPUT_SCHEMA.md) lists the required files, years and fields. It also describes the other panel inputs, including the green credit compilation used for the firm-year list and fallback industry codes, not for credit measures.

## What the repository includes

`data/public/` contains the authors' classifications of seven official 2017 R&D guides and links to the source documents. The classifications do not contain company records, amounts or patent outcomes. The aggregate reference tables are in `results/reference/paper_tables.json`.

The repository does not distribute restricted source records, the prepared firm-year panel, or the industry-year measures derived from those inputs. The guide regressions still require the prepared panel. The title-record analyses require both the CSMAR exports and the authors' classifications linked to those records; neither is included.
