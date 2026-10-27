"""
SimSG - clean the seven source workbooks into tidy CSV tables.

Put this file in the project folder, next to data/, then run:
    python clean_data.py

Folder layout:
    data/raw/         original downloads (not used by this script)
    data/cleaned/     the seven Cleaned_*.xlsx workbooks  -> read, never modified
    data/processed/   tidy CSV tables                     -> written by this script
    data/checks/      cleaning_report.txt, published_area_totals.csv

Every output table is in long format: one row per combination, one count column.
"""

import re
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "sheets"
RAW = DATA / "cleaned"
CLEAN = DATA / "processed"
CHECKS = DATA / "checks"

FILES = {
    "ethnic": "Cleaned_2025_Resident_Area_EthnicGroup_Gender.xlsx",
    "age": "Cleaned_2025_Resident_Area_AgeGroup_Gender.xlsx",
    "area_income": "Cleaned_2025_Household_Area_MonthlyIncome.xlsx",
    "income_dwelling": "Cleaned_2025_Household_MonthlyIncome_DwellingType.xlsx",
    "area_dwelling": "Cleaned_2025_Household_Area_DwellingType.xlsx",
    "spend": "Cleaned_2023_Resident_AverageExpenditure_DwellingType.xlsx",
    "size": "Cleaned_2025_Household_Area_HouseholdSize_Tenancy.xlsx",
}

DROP_AREAS = {"Total", "Others"}  # national total row; household-only grouping of 26 small areas

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# Label standardisation (the single mapping used by every table)
# ---------------------------------------------------------------------------

def strip_footnote(text):
    """'HDB 1- & 2-Room Flats3/' -> 'HDB 1- & 2-Room Flats'; also trims spaces."""
    return re.sub(r"\s*\d+/\s*$", "", str(text)).strip()


DWELLING_KEYS = [
    ("1-2 room", "HDB 1- & 2-Room Flats"),
    ("3 room", "HDB 3-Room Flats"),
    ("4 room", "HDB 4-Room Flats"),
    ("5 room/exec", "HDB 5-Room & Executive Flats"),
    ("condo", "Condominiums & Other Apartments"),
    ("landed", "Landed Properties"),
]
DWELLING_ORDER = [k for k, _ in DWELLING_KEYS]


def dwelling_key(label):
    """Map any spelling of a dwelling type to one short key, or None for totals."""
    s = strip_footnote(label).lower().replace("hdb ", "").replace(" ", "")
    if s in {"total", "dwellings"}:
        return None
    if s.startswith("1-&2-room"):
        return "1-2 room"
    if s.startswith("3-room"):
        return "3 room"
    if s.startswith("4-room"):
        return "4 room"
    if s.startswith("5-room"):
        return "5 room/exec"
    if s.startswith("condominium"):
        return "condo"
    if s.startswith("landed"):
        return "landed"
    raise ValueError(f"Unknown dwelling label: {label!r}")


def income_band(label):
    """'$25,000 -$29,999' / '25,000 - 29,999' -> ('25000-29999', 25000, 29999)."""
    s = re.sub(r"[\s$,]", "", strip_footnote(label))
    if s.lower().startswith("below"):
        hi = int(re.sub(r"\D", "", s))
        return f"0-{hi - 1}", 0, hi - 1
    if s.lower().endswith("&over"):
        lo = int(re.sub(r"\D", "", s))
        return f"{lo}+", lo, None
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise ValueError(f"Unknown income band: {label!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    return f"{lo}-{hi}", lo, hi


def age_group(label):
    """'0 - 4' -> ('0-4', 0); '90 & Over' -> ('90+', 90)."""
    s = re.sub(r"\s", "", str(label))
    if s.lower().endswith("&over"):
        lo = int(re.sub(r"\D", "", s))
        return f"{lo}+", lo
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise ValueError(f"Unknown age group: {label!r}")
    return f"{m.group(1)}-{m.group(2)}", int(m.group(1))


def to_number(series):
    """'-' means nil in these tables, so it becomes 0. Anything else unreadable stays missing."""
    s = series.replace({"-": 0, "–": 0})
    return pd.to_numeric(s, errors="coerce")


def read(key, sheet=0):
    return pd.read_excel(RAW / FILES[key], sheet_name=sheet)


def tidy_areas(df):
    df = df.rename(columns={df.columns[0]: "area"})
    df["area"] = df["area"].astype(str).str.strip()
    return df[~df["area"].isin(DROP_AREAS)].copy()


# ---------------------------------------------------------------------------
# 1. Residents: area x sex x ethnic group
# ---------------------------------------------------------------------------

def clean_ethnic():
    parts = []
    for group in ["Chinese", "Malay", "Indian", "Others"]:
        df = tidy_areas(read("ethnic", group)).drop(columns=["Total"])
        df = df.melt(id_vars="area", var_name="sex", value_name="residents")
        df["ethnic_group"] = group
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["residents"] = to_number(out["residents"])
    return out[["area", "sex", "ethnic_group", "residents"]]


# ---------------------------------------------------------------------------
# 2. Residents: area x age group x sex
# ---------------------------------------------------------------------------

def clean_age():
    parts = []
    for tab, sex in [("Males", "Male"), ("Females", "Female")]:
        df = tidy_areas(read("age", tab)).drop(columns=["Total"])
        df = df.melt(id_vars="area", var_name="age_raw", value_name="residents")
        df["sex"] = sex
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out[["age_group", "age_lo"]] = out["age_raw"].apply(lambda x: pd.Series(age_group(x)))
    out["residents"] = to_number(out["residents"])
    return out[["area", "age_group", "age_lo", "sex", "residents"]]


# ---------------------------------------------------------------------------
# 3. Households: area x monthly household market income
# ---------------------------------------------------------------------------

def clean_area_income():
    df = tidy_areas(read("area_income"))
    published = df.set_index("area")["Total"]
    df = df.drop(columns=["Total"]).melt(id_vars="area", var_name="band_raw", value_name="households")
    df[["income_band", "income_lo", "income_hi"]] = df["band_raw"].apply(lambda x: pd.Series(income_band(x)))
    df["households"] = to_number(df["households"])
    return df[["area", "income_band", "income_lo", "income_hi", "households"]], published


# ---------------------------------------------------------------------------
# 4. Households: income band x dwelling (national seed)
# ---------------------------------------------------------------------------

def clean_income_dwelling():
    df = read("income_dwelling")
    df = df.rename(columns={df.columns[0]: "band_raw"})
    df = df[df["band_raw"].astype(str).str.strip() != "Total"].drop(columns=["Total"])
    df = df.melt(id_vars="band_raw", var_name="dwelling_raw", value_name="households")
    df[["income_band", "income_lo", "income_hi"]] = df["band_raw"].apply(lambda x: pd.Series(income_band(x)))
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["households"] = to_number(df["households"])
    return df[["income_band", "income_lo", "income_hi", "dwelling", "households"]]


# ---------------------------------------------------------------------------
# 5. Households: area x dwelling
# ---------------------------------------------------------------------------

def clean_area_dwelling():
    df = tidy_areas(read("area_dwelling"))
    published = df.set_index("area")["Total"]
    df = df.drop(columns=["Total"]).melt(id_vars="area", var_name="dwelling_raw", value_name="households")
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["households"] = to_number(df["households"])
    return df[["area", "dwelling", "households"]], published


# ---------------------------------------------------------------------------
# 6. Spending per household member x dwelling (HES 2023, S$ per month)
# ---------------------------------------------------------------------------

def clean_spend():
    df = read("spend")
    df = df.rename(columns={df.columns[0]: "category"})
    df["category"] = df["category"].map(strip_footnote)
    df = df.drop(columns=["Total"]).melt(id_vars="category", var_name="dwelling_raw", value_name="spend_per_member")
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["spend_per_member"] = to_number(df["spend_per_member"])
    return df[["dwelling", "category", "spend_per_member"]]


# ---------------------------------------------------------------------------
# 7. Households: dwelling x household size (national), with tenure split
# ---------------------------------------------------------------------------

SIZE_TABS = {"1_person": 1, "2_person": 2, "3_person": 3, "4_person": 4,
             "5_person": 5, "6_people_or_more": 6}  # 6 means "6 or more"


def clean_size():
    parts = []
    for tab, size in SIZE_TABS.items():
        df = read("size", tab)
        df = df.rename(columns={df.columns[0]: "dwelling_raw",
                                "Total": "households", "Owner-Occupied": "owner_occupied", "Rented": "rented"})
        df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
        df = df[df["dwelling"].notna()].copy()  # drops 'Total' and 'HDB Dwellings' subtotal rows
        df["household_size"] = size
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    for c in ["households", "owner_occupied", "rented"]:
        out[c] = to_number(out[c])
    return out[["dwelling", "household_size", "households", "owner_occupied", "rented"]]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run_checks(t, published_income, published_dwelling):
    results = []
    areas = {k: set(t[k]["area"]) for k in ["ethnic", "age", "area_income", "area_dwelling"]}
    ref = areas["area_dwelling"]
    results.append(check("Same areas in all four area tables",
                         all(a == ref for a in areas.values()), f"{len(ref)} areas"))

    for k in ["income_dwelling", "area_dwelling", "spend", "size"]:
        got = set(t[k]["dwelling"])
        results.append(check(f"Dwelling labels match in {k}", got == set(DWELLING_ORDER), str(sorted(got))))

    b1, b2 = set(t["area_income"]["income_band"]), set(t["income_dwelling"]["income_band"])
    results.append(check("Income bands match between area x income and income x dwelling",
                         b1 == b2, f"{len(b1)} bands"))

    for k, col in [("ethnic", "residents"), ("age", "residents"), ("area_income", "households"),
                   ("income_dwelling", "households"), ("area_dwelling", "households"),
                   ("spend", "spend_per_member"), ("size", "households")]:
        n = int(t[k][col].isna().sum())
        neg = int((t[k][col] < 0).sum())
        results.append(check(f"No missing or negative values in {k}", n == 0 and neg == 0,
                             f"missing={n}, negative={neg}"))

    e = t["ethnic"].groupby(["area", "sex"])["residents"].sum()
    a = t["age"].groupby(["area", "sex"])["residents"].sum()
    gap = (e - a).abs().max()
    results.append(check("Area x sex totals agree between ethnic and age tables (rounding <= 50)",
                         gap <= 50, f"max gap {gap:.0f}"))

    inc = t["area_income"].groupby("area")["households"].sum()
    dwe = t["area_dwelling"].groupby("area")["households"].sum()
    g1 = (inc - published_income).abs().max()
    g2 = (dwe - published_dwelling).abs().max()
    results.append(check("Area income rows close to published area totals (rounding <= 500)",
                         g1 <= 500, f"max gap {g1:.0f}"))
    # The published area total also counts households in dwelling types that have no column
    # (about 3,500 nationally), so the six dwelling columns can fall short of it.
    short = published_dwelling - dwe
    log(f"[INFO] Households in unlisted dwelling types per area: up to {short.max():.0f} "
        f"(largest: {short.idxmax()}); total across 29 areas {short.sum():.0f}")
    results.append(check("Dwelling rows never exceed the published area total beyond rounding (<= 200)",
                         short.min() >= -200, f"min {short.min():.0f}"))
    results.append(check("Published area totals identical in income and dwelling tables",
                         (published_income - published_dwelling).abs().max() == 0))

    nat_size = t["size"].groupby("dwelling")["households"].sum()
    nat_dw = t["area_dwelling"].groupby("dwelling")["households"].sum()
    log(f"[INFO] Size table national households: {int(nat_size.sum()):,}; "
        f"29-area dwelling table: {int(nat_dw.sum()):,} (difference is the dropped 'Others' row + rounding)")

    fnb = t["spend"].query("category == 'Food and Beverage Serving Services'")
    results.append(check("Spending table has Food and Beverage Serving Services for all 6 dwellings",
                         len(fnb) == 6))
    return all(results)


# ---------------------------------------------------------------------------

def main():
    missing = [f for f in FILES.values() if not (RAW / f).exists()]
    if missing:
        raise SystemExit(f"Not found in {RAW}:\n  " + "\n  ".join(missing))
    CLEAN.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    tables = {}
    tables["ethnic"] = clean_ethnic()
    tables["age"] = clean_age()
    tables["area_income"], published_income = clean_area_income()
    tables["income_dwelling"] = clean_income_dwelling()
    tables["area_dwelling"], published_dwelling = clean_area_dwelling()
    tables["spend"] = clean_spend()
    tables["size"] = clean_size()

    names = {
        "ethnic": "resident_area_sex_ethnic.csv",
        "age": "resident_area_age_sex.csv",
        "area_income": "household_area_income.csv",
        "income_dwelling": "household_income_dwelling.csv",
        "area_dwelling": "household_area_dwelling.csv",
        "spend": "spend_per_member_dwelling.csv",
        "size": "household_dwelling_size.csv",
    }
    for k, df in tables.items():
        df.to_csv(CLEAN / names[k], index=False)
        log(f"Wrote data/processed/{names[k]}  ({len(df)} rows)")

    # Keep the published totals as a reference file (used for checks, never for sampling).
    res = read("ethnic", "Total").rename(columns=lambda c: c.strip())
    res = res.rename(columns={res.columns[0]: "area"})
    res["area"] = res["area"].astype(str).str.strip()
    res = res.set_index("area")
    ref = pd.DataFrame({
        "households_published": published_dwelling,
        "households_in_6_dwelling_types": tables["area_dwelling"].groupby("area")["households"].sum(),
        "residents_published": res.loc[published_dwelling.index, "Total"],
        "residents_male": res.loc[published_dwelling.index, "Male"],
        "residents_female": res.loc[published_dwelling.index, "Female"],
    })
    ref.index.name = "area"
    ref.to_csv(CHECKS / "published_area_totals.csv")
    log("Wrote data/checks/published_area_totals.csv  (reference only)")

    log("")
    ok = run_checks(tables, published_income, published_dwelling)
    log("")
    log("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - fix before building the fit")
    (CHECKS / "cleaning_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()