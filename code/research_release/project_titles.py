from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import warnings
import numpy as np
import openpyxl
import pandas as pd
from .restricted_rules import MissingRules

TITLE_RULES = MissingRules("project_titles.TITLE_RULES")

FIRM, AGRI, INDUSTRY = "\u8bc1\u5238\u4ee3\u7801", "\u662f\u5426\u6d89\u519c", "industry_code_std"

SOURCE_FIELDS = {2: "ExpenseResearch", 5: "CapitalRDExpenditure"}

PATTERNS = {
    "pollution_control": (
        r"\u6c61\u6c34|\u5e9f\u6c34|\u5e9f\u6c14|\u56fa\u5e9f|\u5371\u5e9f|\u5371\u9669\u5e9f|\u5e9f\u5f03\u7269|\u5783\u573e|\u6c61\u6ce5|\u4e09\u5e9f|"
        r"\u8131\u786b|\u8131\u785d|\u9664\u5c18|\u964d\u5c18|\u6c61\u67d3|\u5c3e\u6c14|\u4f4e\u6c2e|\u8d85\u4f4e\u6392\u653e|\u96f6\u6392\u653e|"
        r"\u70df\u6c14|\u6e05\u6d01\u751f\u4ea7|\u6e05\u6d01\u5236\u9769|\u6e05\u6d01\u5316\u751f\u7269\u9176|(?:SCR|SNCR)|"
        r"(?:\u571f\u58e4|\u6c34\u4f53|\u751f\u6001).{0,6}\u4fee\u590d|\u6c34\u5904\u7406|\u4e2d\u6c34\u56de\u7528|\u518d\u751f\u6c34"
    ),
    "energy_or_water_efficiency": (
        r"\u8282\u80fd|\u8282\u6c34|\u8282\u7535|\u964d\u8017|\u80fd\u6548|\u70ed\u6548\u7387|\u4f59\u70ed|\u5e9f\u70ed|\u51b7\u51dd\u70ed\u56de\u6536|"
        r"\u4f4e\u80fd\u8017|\u4f4e\u8017|\u80fd\u8017\u964d\u4f4e|\u80fd\u8017\u7279\u6027|\u8282\u80fd\u51cf\u6392"
    ),
    "renewable_energy_and_electrification": (
        r"\u5149\u4f0f|\u592a\u9633\u80fd|\u98ce\u7535|\u98ce\u529b\u53d1\u7535|\u53ef\u518d\u751f\u80fd\u6e90|\u6e05\u6d01\u80fd\u6e90|\u751f\u7269\u8d28|\u6cbc\u6c14|"
        r"\u6c22\u80fd|\u71c3\u6599\u7535\u6c60|\u7535\u52a8\u6c7d\u8f66|\u65b0\u80fd\u6e90\u6c7d\u8f66|\u7eaf\u7535\u52a8|\u50a8\u80fd|\u52a8\u529b\u7535\u6c60|"
        r"\u63ba\u6c22|\u65b0\u80fd\u6e90.{0,5}(?:\u6c7d\u8f66|\u7535\u6c60|\u53d1\u7535|\u5e76\u7f51|\u8bbe\u5907|\u7535\u673a|\u7535\u63a7|\u53d8\u901f\u7bb1)"
    ),
    "waste_resource_recovery": (
        r"\u8d44\u6e90\u5316|\u5faa\u73af\u5229\u7528|\u518d\u751f\u8d44\u6e90|\u518d\u5236\u9020|\u5e9f\u65e7|"
        r"\u5e9f.{0,10}(?:\u56de\u6536|\u7efc\u5408\u5229\u7528)|\u5c3e\u77ff.{0,10}(?:\u5229\u7528|\u56de\u6536)|"
        r"(?:\u6eb6\u5242|\u6eb6\u5a92|\u8680\u523b\u6db2|\u5e9f\u9178|\u5e9f\u7802\u6d46|\u50ac\u5316\u5242|\u78b1\u6db2|\u5e9f\u6599|\u5c3e\u7802|\u5c3e\u6e23|\u4e8c\u6b21\u8d44\u6e90).{0,8}\u56de\u6536|"
        r"\u56de\u6536.{0,8}(?:\u5e9f|\u4f59\u70ed|\u76d0\u9178|\u50ac\u5316\u5242)"
    ),
    "climate_or_environment_monitoring": (
        r"\u78b3\u6355|\u78b3\u51cf\u6392|\u51cf\u78b3|(?:\u78b3\u6392\u653e|\u6e29\u5ba4\u6c14\u4f53).{0,12}(?:\u76d1\u6d4b|\u63a7\u5236|\u68c0\u6d4b)|"
        r"(?:\u751f\u6001|\u6c34\u8d28|\u6c34\u73af\u5883|\u6d77\u6d0b\u73af\u5883|\u5927\u6c14|\u7a7a\u6c14\u8d28\u91cf|\u7a7a\u6c14\u73af\u5883).{0,12}(?:\u76d1\u6d4b|\u68c0\u6d4b|\u6cbb\u7406|\u4fee\u590d|\u4fdd\u62a4)"
    ),
}

BROAD_GREEN = (
    r"\u73af\u4fdd|\u73af\u5883|\u7eff\u8272|\u4f4e\u78b3|\u6e05\u6d01|\u65b0\u80fd\u6e90|\u7535\u6c60|\u5faa\u73af|\u56de\u6536|\u518d\u751f|"
    r"\u6838\u7535|\u6838\u80fd|\u6c22|\u8d44\u6e90\u7efc\u5408\u5229\u7528"
)

TECHNICAL_OBJECT = (
    r"\u7814\u53d1|\u7814\u7a76|\u5f00\u53d1|\u7814\u5236|\u6280\u672f|\u5de5\u827a|\u7cfb\u7edf|\u8bbe\u5907|\u88c5\u5907|\u88c5\u7f6e|\u6750\u6599|"
    r"\u4ea7\u54c1|\u673a\u68b0|\u63a7\u5236|\u6027\u80fd|\u4f18\u5316|\u6539\u8fdb|\u8f6f\u4ef6|\u50ac\u5316|\u50ac\u5316\u5242|\u819c|"
    r"\u7535\u6c60|\u7535\u7f06|\u7535\u673a|\u53d8\u901f\u7bb1|\u76d1\u6d4b|\u68c0\u6d4b|\u9600\u95e8|\u7b97\u6cd5|\u9505\u7089|\u70ed\u6cf5"
)

ORDINARY_COST = (
    r"\u73af\u4fdd\u8d39(?:\u7528)?|\u73af\u4fdd\u652f\u51fa|\u73af\u5883\u4fdd\u62a4\u8d39|\u6392\u6c61\u8d39|\u6392\u6c61\u6743|\u78b3\u6392\u653e\u8d44\u683c|"
    r"\u8ba4\u8bc1\u652f\u51fa|\u6e05\u6d01\u8d39|\u6e05\u6d17\u8d39|\u7eff\u5316\u8d39|\u7269\u4e1a|\u6c34\u7535\u8d39|\u5382\u533a\u73af\u5883|"
    r"\u5783\u573e\u6e05\u8fd0|\u5783\u573e\u6e05\u7406|\u5e9f\u7269\u5904\u7f6e\u8d39|\u6c61\u6c34\u5904\u7406\u8d39|\u5e9f\u6c34\u5904\u7406\u8d39"
)

MIXED_CONTEXT = (
    r"\u53ca\u5176\u4ed6|\u4ee5\u53ca\u5176\u4ed6|\u7b49\u8d39\u7528|\u7ba1\u7406\u8d39|\u5382\u623f|\u5236\u9020\u57fa\u5730|\u751f\u4ea7\u57fa\u5730|"
    r"\u751f\u4ea7\u7ebf\u5efa\u8bbe|\u7814\u53d1\u4e2d\u5fc3\u5efa\u8bbe|\u7814\u53d1\u57fa\u5730\u5efa\u8bbe|\u7814\u7a76\u4e2d\u5fc3\u5efa\u8bbe|"
    r"\u4f17\u521b\u7a7a\u95f4|\u4ea7\u4e1a\u5b75\u5316|\u4ea7\u4e1a\u5316\u57fa\u5730|\u5b9e\u8bad|\u793a\u8303\u56ed"
)

NONENV_CONTEXT = (
    r"\u5faa\u73af\u80bf\u7624|\u8840\u6db2\u51c0\u5316|\u8840\u6db2\u900f\u6790|\u8840\u6db2\u5faa\u73af|\u4f53\u5916\u5faa\u73af|\u7ec6\u80deDNA|"
    r"\u5f00\u53d1\u73af\u5883|\u4e91\u73af\u5883|\u8ba1\u7b97\u73af\u5883|\u8bd5\u9a8c\u73af\u5883|\u7535\u78c1\u73af\u5883|\u8f66\u8def\u534f\u540c\u73af\u5883|"
    r"\u4f4e\u78b3\u9187|\u4f4e\u78b3\u5316\u5408\u7269|\u6e05\u6d01\u5ea6|\u8868\u9762\u6e05\u6d01|\u6d01\u51c0\u5ba4|"
    r"\u9ad8\u80fd\u91cf\u5851\u6599\u5916\u58f3\u9502\u79bb\u5b50\u7535\u6c60|\u9502\u79bb\u5b50\u7535\u6c60\u901a\u7528\u6a21\u5757"
)

def normalise(value: str | None) -> str:
    return "" if value is None or pd.isna(value) else re.sub(r"\s+", "", str(value))

def classify(value: str | None) -> tuple[str, str, str]:
    text = normalise(value)
    if not text:
        return "uncertain_or_mixed_title", "missing_title", "Project purpose is unavailable."
    if text in TITLE_RULES:
        category, reason = TITLE_RULES[text]
        return category, "reviewed_title_rule", reason
    if re.search(NONENV_CONTEXT, text):
        return "uncertain_or_mixed_title", "nonenvironmental_word_context", (
            "A green-screen word has a nonenvironmental or ambiguous technical meaning."
        )
    theme = next((name for name, pattern in PATTERNS.items() if re.search(pattern, text)), None)
    broad = bool(re.search(BROAD_GREEN, text))
    if re.search(ORDINARY_COST, text):
        if re.search(r"\u7814\u53d1|\u7814\u7a76|\u5f00\u53d1|\u7814\u5236|\u6280\u672f|\u5de5\u827a", text):
            return "uncertain_or_mixed_title", "mixed_cost_and_rd", "Cost and R&D purposes are not separately identified."
        return "routine_environmental_cost", "routine_cost", "Routine environmental or premises cost, not a defined green R&D project."
    if re.search(MIXED_CONTEXT, text) and (theme or broad):
        return "uncertain_or_mixed_title", "mixed_or_infrastructure", (
            "Facilities or other uses cannot be separated from green R&D."
        )
    if theme and re.search(TECHNICAL_OBJECT, text):
        return "clear_green_rd_title", theme, (
            "Title explicitly links an environmental/clean-energy objective to a technical project in an R&D accounting category."
        )
    if theme or broad:
        return "uncertain_or_mixed_title", "insufficient_objective_or_technology", (
            "Title suggests a green theme but does not sufficiently define an environmental technical project."
        )
    return "no_explicit_green_purpose", "no_explicit_green_purpose", (
        "Title does not establish a green purpose; this is not evidence of zero true green R&D."
    )

def read_export(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book.active
        sheet.reset_dimensions()
        rows = sheet.iter_rows(values_only=True)
        headers = next(rows)
        next(rows)
        next(rows)
        frame = pd.DataFrame(list(rows), columns=headers)
        frame["source_excel_row"] = np.arange(4, len(frame) + 4)
    finally:
        book.close()
    if frame.columns.duplicated().any() or frame.Symbol.isna().any():
        raise ValueError("Invalid export headers or stock codes.")
    symbols = frame.Symbol.astype(str)
    if not symbols.str.fullmatch(r"\d{1,6}").all():
        raise ValueError("Unexpected stock-code format.")
    frame[FIRM] = symbols.str.zfill(6)
    dates = pd.to_datetime(frame.EndDate, errors="raise")
    if not (dates.dt.month.eq(12) & dates.dt.day.eq(31)).all():
        raise ValueError("Non-annual export records.")
    frame["year"] = dates.dt.year.astype(int)
    return frame

def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))

def subset_summary(frame: pd.DataFrame) -> dict:
    rows = []
    for label, value in [("All", None), ("Agricultural", 1), ("Non-agricultural", 0)]:
        part = frame if value is None else frame.loc[frame[AGRI].eq(value)]
        rows.append({
            "group": label, "observations": len(part), "firms": int(part[FIRM].nunique()),
            "industries": int(part[INDUSTRY].nunique()),
            "industry_year_cells": len(part[[INDUSTRY, "year"]].drop_duplicates()),
            "first_year": int(part.year.min()) if len(part) else None,
            "last_year": int(part.year.max()) if len(part) else None,
        })
    return {"groups": rows, "by_year": records(frame.groupby("year").agg(
        observations=(FIRM, "size"), agricultural_observations=(AGRI, "sum"),
        industries=(INDUSTRY, "nunique"),
    ).reset_index())}

def support_coverage(flags: pd.DataFrame, panel: pd.DataFrame, common: pd.DataFrame) -> dict:
    if flags.duplicated([FIRM, "year"]).any():
        raise ValueError("Duplicate source firm-year flag.")
    source = flags.merge(panel[[FIRM, "year", AGRI, INDUSTRY]], on=[FIRM, "year"],
                         how="inner", validate="one_to_one")
    source = source.loc[source[AGRI].eq(0) & source[INDUSTRY].str.fullmatch(r"[A-Z]\d{2}")].copy()
    cells = source.groupby([INDUSTRY, "year"]).agg(
        recorded_firms=(FIRM, "size"), clear_firms=("clear", "sum"),
        amount_eligible_firms=("amount_eligible", "sum"),
    ).reset_index().rename(columns={"year": "source_year"})
    lag = cells.assign(year=cells.source_year + 1)
    matched = common.merge(lag, on=[INDUSTRY, "year"], how="left", validate="many_to_one", sort=False)
    own = source[[FIRM, INDUSTRY, "year", "clear", "amount_eligible"]].rename(
        columns={"clear": "focal_clear", "amount_eligible": "focal_amount_eligible"})
    own["year"] += 1
    matched = matched.merge(own, on=[FIRM, INDUSTRY, "year"], how="left", validate="one_to_one", sort=False)
    focal = matched[AGRI].eq(0) & matched.focal_clear.notna()
    matched["effective_recorded"] = matched.recorded_firms - focal.astype(int)
    matched["effective_clear"] = matched.clear_firms - matched.focal_clear.fillna(False).astype(int).where(focal, 0)
    matched["effective_amount"] = matched.amount_eligible_firms - matched.focal_amount_eligible.fillna(False).astype(int).where(focal, 0)
    assert matched[[FIRM, "year"]].equals(common[[FIRM, "year"]].reset_index(drop=True))
    assert matched.loc[matched.recorded_firms.notna(), "year"].eq(
        matched.loc[matched.recorded_firms.notna(), "source_year"] + 1).all()
    assert (matched.effective_clear.dropna() <= matched.effective_recorded.dropna()).all()
    sufficient = matched.effective_recorded.ge(3)
    choices = {
        "at_least_1_clear_peer": matched.effective_clear.ge(1),
        "at_least_3_clear_peers": matched.effective_clear.ge(3),
        "at_least_3_complete_amount_peers": matched.effective_amount.ge(3),
        "at_least_3_recorded_peers": sufficient,
        "recorded_peers_and_at_least_1_clear": sufficient & matched.effective_clear.ge(1),
    }
    result = {
        "benchmark_nonag_firm_years": len(source),
        "benchmark_nonag_clear_firm_years": int(source.clear.sum()),
        "benchmark_nonag_industries": int(source[INDUSTRY].nunique()),
        "benchmark_cells": records(cells),
        "coverage": {label: subset_summary(matched.loc[mask]) for label, mask in choices.items()},
        "supported_recorded_cells_without_clear_title": len(
            matched.loc[sufficient & matched.effective_clear.eq(0), [INDUSTRY, "year"]].drop_duplicates()),
        "focal_removals": int(focal.sum()),
        "matched_key_sha256": file_key_hash(matched.loc[sufficient]),
        "rule": "Strict previous calendar year, detailed source industry, non-agricultural peers only, focal peer excluded before support count. Three peers is inherited feasibility screen, not adequate-inference guarantee.",
    }
    return result

def file_key_hash(frame: pd.DataFrame) -> str:
    rows = frame[[FIRM, "year"]].sort_values([FIRM, "year"])
    return hashlib.sha256(rows.to_csv(index=False, lineterminator="\n").encode()).hexdigest()

def make_flags(frame: pd.DataFrame, total: pd.DataFrame, source_code: int) -> pd.DataFrame:
    recorded = frame.loc[frame.normalized_title.ne(""), [FIRM, "year"]].drop_duplicates()
    clear = frame.loc[frame.category.eq("clear_green_rd_title")]
    groups = clear.groupby([FIRM, "year"]).agg(
        projects=("Item", "size"), known=("Amount", "count"),
        amount_sum=("Amount", lambda s: s.sum(min_count=1)),
        negative=("Amount", lambda s: int(s.lt(0).sum())),
        duplicate_title=("duplicate_normalized_title", "max"),
        accounting_adjustment=("accounting_adjustment_title", "max"),
    ).reset_index()
    denominator = "GrossRevenue" if source_code == 2 else "TotalAssets"
    groups = groups.merge(total[[FIRM, "year", denominator]], on=[FIRM, "year"],
                          how="left", validate="one_to_one")
    # Eligibility is only an availability diagnostic; no monetary environment or regression is estimated.
    groups["amount_eligible"] = (
        groups.projects.eq(groups.known) & groups.negative.eq(0)
        & ~groups.duplicate_title & ~groups.accounting_adjustment & groups[denominator].gt(0))
    groups["clear"] = True
    flags = recorded.merge(groups[[FIRM, "year", "clear", "amount_eligible"]],
                           on=[FIRM, "year"], how="left", validate="one_to_one")
    flags[["clear", "amount_eligible"]] = flags[["clear", "amount_eligible"]].fillna(False).astype(bool)
    return flags

def self_test() -> None:
    cases = {
        "\u9ad8\u542b\u76d0\u5e9f\u6c34\u4e0e\u53cd\u6e17\u900f\u819c\u6d53\u6c34\u5904\u7406\u6280\u672f": "clear_green_rd_title",
        "\u65b0\u80fd\u6e90\u6c7d\u8f66\u7535\u6c60\u7ba1\u7406\u7cfb\u7edf": "clear_green_rd_title",
        "\u8680\u523b\u6db2\u65e0\u5bb3\u5316\u53ca\u8d44\u6e90\u5316\u7684\u5de5\u827a\u7814\u7a76\u9879\u76ee": "clear_green_rd_title",
        "\u73af\u4fdd\u8d39": "routine_environmental_cost",
        "\u7269\u4e1a\u7ba1\u7406\u8d39\u3001\u6e05\u6d01\u8d39\u3001\u6c34\u7535\u8d39": "routine_environmental_cost",
        "\u4eba\u5faa\u73af\u80bf\u7624DNA\u591a\u57fa\u56e0\u7a81\u53d8\u8054\u5408\u68c0\u6d4b\u8bd5\u5242\u76d2": "uncertain_or_mixed_title",
        "\u542b\u7fb0\u57fa\u7684\u4f4e\u78b3\u5316\u5408\u7269\u52a0\u6c22\u50ac\u5316\u5242\u7684\u5f00\u53d1\u4e0e\u5e94\u7528": "uncertain_or_mixed_title",
        "\u73af\u4fdd\u7cfb\u7edf\u53ca\u5176\u4ed6": "uncertain_or_mixed_title",
        "\u901a\u4fe1\u673a\u623f\u73af\u5883\u76d1\u6d4b\u8f6f\u4ef6": "uncertain_or_mixed_title",
        "\u4e00\u79cd\u73af\u5883\u5149\u5f3a\u5ea6\u68c0\u6d4b\u65b9\u6cd5": "uncertain_or_mixed_title",
        "\u4ece\u542b\u9517\u591a\u91d1\u5c5e\u7269\u6599\u4e2d\u56de\u6536\u9517\u7684\u5de5\u827a\u7814\u7a76": "uncertain_or_mixed_title",
        "\u79fb\u52a8\u5e94\u7528\u9690\u79c1\u5408\u89c4\u81ea\u52a8\u5316\u68c0\u6d4b\u5e73\u53f0V1.0": "no_explicit_green_purpose",
    }
    for title, category in cases.items():
        assert classify(title)[0] == category, title
    firms = ["000001", "000002", "000003", "000004", "000005"]
    panel = pd.DataFrame({
        FIRM: firms * 3, "year": np.repeat([2010, 2011, 2013], 5),
        AGRI: [0, 0, 0, 1, 1] * 3, INDUSTRY: ["C01"] * 15,
    })
    flags = pd.DataFrame({
        FIRM: firms[:4], "year": [2010] * 4,
        "clear": [True, False, True, True],
        "amount_eligible": [True, False, False, True],
    })
    common = panel.loc[panel.year.isin([2011, 2013])].copy()
    test = support_coverage(flags, panel, common)
    all_count = test["coverage"]["at_least_3_recorded_peers"]["groups"][0]["observations"]
    assert all_count == 2  # Focal exclusion leaves only the two agricultural observations.
    assert test["coverage"]["at_least_3_clear_peers"]["groups"][0]["observations"] == 0
    assert test["coverage"]["at_least_1_clear_peer"]["groups"][0]["observations"] == 5
    assert len(test["benchmark_cells"]) == 1
    assert test["benchmark_cells"][0]["clear_firms"] == 2
    gap = support_coverage(flags.assign(year=2009), panel, common)
    assert gap["coverage"]["at_least_1_clear_peer"]["groups"][0]["observations"] == 0
    transitions = common.copy()
    transitions.loc[transitions[FIRM].eq("000001"), INDUSTRY] = "C02"
    changed = support_coverage(flags, panel, transitions)
    assert changed["focal_removals"] == 2
    assert changed["coverage"]["at_least_1_clear_peer"]["groups"][0]["observations"] == 4
    projects = pd.DataFrame({
        FIRM: ["000001", "000002", "000003", "000003", "000004"],
        "year": [2010] * 5, "Item": ["test"] * 5,
        "normalized_title": ["test"] * 5,
        "category": ["clear_green_rd_title"] * 5,
        "Amount": [0., np.nan, 1., 2., -1.],
        "duplicate_normalized_title": [False, False, True, True, False],
        "accounting_adjustment_title": [False] * 5,
    })
    denominators = pd.DataFrame({FIRM: firms[:4], "year": [2010] * 4,
                                 "GrossRevenue": [100.] * 4})
    availability = make_flags(projects, denominators, 2)
    assert availability["clear"].all()
    assert availability["amount_eligible"].tolist() == [True, False, False, False]
    impaired = make_flags(projects.assign(accounting_adjustment_title=True), denominators, 2)
    assert not impaired["amount_eligible"].any()
