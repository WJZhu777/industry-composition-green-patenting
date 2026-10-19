from __future__ import annotations

import math
from pathlib import Path
import re
import openpyxl
import pandas as pd
from .restricted_rules import MissingRules

TITLE_REVIEW_RULES = MissingRules("subsidy_titles.TITLE_REVIEW_RULES")

FIRM, AGRI, INDUSTRY = "\u8bc1\u5238\u4ee3\u7801", "\u662f\u5426\u6d89\u519c", "industry_code_std"

RAW_RD = r"\u7814\u53d1|\u7814\u7a76\u5f00\u53d1|\u7814\u7a76\u4e0e\u5f00\u53d1|\u6280\u672f\u5f00\u53d1|\u6280\u672f\u7814\u7a76"

RAW_GREEN = (
    r"\u73af\u4fdd|\u73af\u5883|\u6c61\u67d3|\u6c61\u6c34|\u5e9f\u6c34|\u5e9f\u6c14|\u5e9f\u7269|\u5e9f\u5f03|\u56fa\u5e9f|\u5783\u573e|\u8131\u786b|\u8131\u785d|\u8282\u80fd|\u51cf\u6392|"
    r"\u6e05\u6d01\u80fd\u6e90|\u65b0\u80fd\u6e90|\u53ef\u518d\u751f|\u5149\u4f0f|\u98ce\u7535|\u592a\u9633\u80fd|\u751f\u7269\u8d28|\u71c3\u6599\u7535\u6c60|\u52a8\u529b.{0,5}\u7535\u6c60|"
    r"\u78f7\u5438\u9644|\u78b3\u6355|\u4f4e\u78b3|\u6c34\u5904\u7406|\u7eff\u8272"
)

SPECIFIC_GREEN = (
    r"\u73af\u4fdd|\u73af\u5883\u4fdd\u62a4|\u73af\u5883\u6cbb\u7406|\u73af\u5883\u53cb\u597d|\u73af\u5883\u76d1\u6d4b|\u73af\u5883\u81ea\u52a8\u76d1\u6d4b|\u751f\u6001\u73af\u5883|\u6c61\u67d3|"
    r"\u6c61\u6c34|\u5e9f\u6c34|\u5e9f\u6c14|\u5e9f\u7269|\u5e9f\u5f03|\u56fa\u5e9f|\u5783\u573e|\u8131\u786b|\u8131\u785d|\u8282\u80fd|\u51cf\u6392|\u6e05\u6d01\u80fd\u6e90|"
    r"\u65b0\u80fd\u6e90|\u53ef\u518d\u751f|\u5149\u4f0f|\u98ce\u7535|\u592a\u9633\u80fd|\u751f\u7269\u8d28|\u71c3\u6599\u7535\u6c60|\u52a8\u529b.{0,5}\u7535\u6c60|"
    r"\u78f7\u5438\u9644|\u78b3\u6355|\u4f4e\u78b3|\u6c34\u5904\u7406"
)

ADMIN_CONTEXT = r"(?:\u7ecf\u6d4e)?\u6280\u672f\u5f00\u53d1\u533a|(?:\u751f\u6001\u73af\u5883|\u73af\u5883\u4fdd\u62a4|\u73af\u4fdd)(?:\u5c40|\u5385|\u90e8\u95e8)"

MIXED_PURPOSE = (
    r"\u7814\u53d1\u4e0e\u73af\u4fdd|\u7814\u53d1\u53ca\u73af\u4fdd|\u73af\u4fdd\u6539\u9020[\u3001,\uff0c].{0,15}(?:\u65b0)?\u6280\u672f\u5f00\u53d1|"
    r"\u7814\u53d1\u4e00\u4f53\u5316|\u6269\u5efa|\u5382\u623f|\u5236\u9020\u57fa\u5730|\u751f\u4ea7\u57fa\u5730|\u751f\u4ea7\u7ebf\u6539\u9020"
)

NONENV_CONTEXT = r"\u4e91\u73af\u5883|\u7f51\u7edc\u73af\u5883|\u8f6f\u4ef6\u73af\u5883|\u5f00\u53d1\u73af\u5883|\u73af\u5883\u5efa\u8bbe|\u8425\u5546\u73af\u5883|\u7ecf\u8425\u73af\u5883|\u5e02\u573a\u73af\u5883"

INFRASTRUCTURE = r"\u7814\u53d1\u4e2d\u5fc3|\u7814\u7a76\u4e2d\u5fc3|\u7814\u53d1\u5e73\u53f0|\u6280\u672f\u7814\u7a76\u4e2d\u5fc3|\u5b9e\u9a8c\u5ba4|\u7814\u53d1\u80fd\u529b|\u7814\u53d1\u8bbe\u5907"

SPECIFIC_GREEN += (
    r"|\u6c61\u6ce5|\u4e09\u5e9f|VOC.{0,6}(?:\u6392\u653e|\u63a7\u5236|\u6cbb\u7406)|\u8d44\u6e90\u6d88\u8017|\u80fd\u6548|\u80fd\u8017|\u8282\u6c34|\u8282\u7535|"
    r"\u56de\u6536|\u5faa\u73af\u5229\u7528|\u8d44\u6e90\u5316|\u518d\u5236\u9020|\u7eaf\u7535\u52a8|\u6e05\u6d01\u751f\u4ea7|\u751f\u6001\u4fee\u590d|\u6c34\u73af\u5883.{0,8}(?:\u68c0\u6d4b|\u76d1\u6d4b|\u4fee\u590d)"
)

NONENV_CONTEXT += r"|\u4e91\u8ba1\u7b97\u73af\u5883|\u4e91\u865a\u62df\u73af\u5883|\u8ba1\u7b97\u73af\u5883|\u7f51\u683c\u73af\u5883|\u5f02\u6784\u73af\u5883|\u975e\u53d7\u63a7\u73af\u5883"

def normalise_title(value: str) -> str:
    return re.sub(r"\s+", "", str(value))

def classify_title(value: str | None) -> tuple[str, str, str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return "uncertain", "missing_title", "No usable project title."
    original = normalise_title(value)
    for phrase, (category, reason) in TITLE_REVIEW_RULES.items():
        if normalise_title(phrase) in original:
            return category, "targeted_title_review", reason
    text = re.sub(ADMIN_CONTEXT, "", original)
    if not re.search(RAW_RD, text):
        return "excluded", "no_explicit_project_rd", "After removing administrative names, the title does not meet the explicit-project-R&D wording rule."
    if re.search(r"\u4e13\u5229|\u77e5\u8bc6\u4ea7\u6743", text):
        return "uncertain", "patent_linked_title", "Patent-related support is not automatically an independent R&D-input measure."
    if re.search(MIXED_PURPOSE, text):
        return "uncertain", "mixed_purpose", "The title mixes R&D with other investments or separate uses."
    if re.search(INFRASTRUCTURE, text) and re.search(r"\u8003\u8bc4|\u7ee9\u6548|\u4e1a\u7ee9|\u8ba4\u5b9a", text):
        return "uncertain", "institution_recognition", "Institution recognition or performance awards do not specify current green R&D project support."
    clauses = re.split(r"[\uff1b;\u3002]", text)
    paired = [
        clause for clause in clauses
        if re.search(RAW_RD, clause) and re.search(SPECIFIC_GREEN, clause)
    ]
    if paired:
        if len(clauses) > 1 and any(
            clause and not (
                re.search(RAW_RD, clause) and re.search(SPECIFIC_GREEN, clause)
            ) for clause in clauses
        ):
            return "uncertain", "multiple_unallocated_projects", "Multiple listed purposes cannot all be assigned to green R&D."
        kind = "green_rd_infrastructure" if re.search(INFRASTRUCTURE, text) else "green_rd_project"
        return "clear_title", kind, "Explicit R&D and environmental/clean-energy content in the same project title."
    if re.search(SPECIFIC_GREEN, text):
        return "uncertain", "separate_rd_and_green_content", "R&D and environmental content are present but not clearly linked."
    if re.search(NONENV_CONTEXT, text):
        return "excluded", "nonenvironmental_context", "The environmental word refers to computing, premises or business context."
    return "uncertain", "generic_green_label", "A generic green label or agency name does not specify environmental R&D."

def read_source(path: Path) -> tuple[pd.DataFrame, dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.active
        sheet.reset_dimensions()
        iterator = sheet.iter_rows(values_only=True)
        header, chinese, units = next(iterator), next(iterator), next(iterator)
        frame = pd.DataFrame(list(iterator), columns=header)
    finally:
        workbook.close()
    needed = {"Symbol", "EndDate", "StateTypeCode", "Item", "Amount", "Currency", "Explanation"}
    if not needed.issubset(frame.columns) or frame.columns.duplicated().any():
        raise ValueError("Invalid source schema.")
    if frame.Symbol.isna().any():
        raise ValueError("Missing stock identifiers.")
    codes = frame.Symbol.astype(str)
    if not codes.str.fullmatch(r"\d{1,6}").all():
        raise ValueError("Unexpected stock-code format.")
    frame[FIRM] = codes.str.zfill(6)
    dates = pd.to_datetime(frame.EndDate, errors="raise")
    if not (dates.dt.month.eq(12) & dates.dt.day.eq(31)).all():
        raise ValueError("Non-annual records need a separate rule.")
    frame["year"] = dates.dt.year
    frame["report_type"] = pd.to_numeric(frame.StateTypeCode, errors="raise")
    if not frame.report_type.isin([1, 2]).all():
        raise ValueError("Unknown financial-report type.")
    frame["numeric_amount"] = pd.to_numeric(frame.Amount, errors="raise")
    if any(not math.isfinite(value) for value in frame.numeric_amount.dropna()):
        raise ValueError("Nonfinite amounts.")
    frame["source_excel_row"] = range(4, len(frame) + 4)
    meta = {
        "source_records_all_years": len(frame),
        "source_year_min": int(frame.year.min()),
        "source_year_max": int(frame.year.max()),
        "amount_unit": units[list(header).index("Amount")],
        "amount_label": chinese[list(header).index("Amount")],
        "missing_titles_all_years": int(frame.Item.isna().sum()),
    }
    return frame, meta
